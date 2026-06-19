import json
import logging
import time
import os
import sys
import random
import threading
import shutil
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import settings
from db_manager import DBManager, SQLQueryResult
from llm_client import BaseLLMClient

logger = logging.getLogger(__name__)

def _normalize_cell(value: Any) -> Tuple[int, Any]:
    """
    Normalize one cell into a (type_tag, comparable_value) pair.

      NULL                 -> (0, "")
      numbers (int/float/bool) -> (1, round(float, 6))   so 1 and 1.0 (and 0.1+0.2 vs 0.3) match
      text / bytes / other -> (2, str(value))

    The type_tag keeps NULLs, numbers and strings in separate ordering bands so
    sorting a column with mixed types never compares a float against a str
    (which would raise TypeError).
    """
    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, float(value))
    if isinstance(value, (int, float)):
        return (1, round(float(value), 6))
    if isinstance(value, bytes):
        return (2, value.decode("utf-8", "replace"))
    return (2, str(value))


def normalize_execution_result(rows: List[Tuple[Any, ...]]) -> str:
    """
    Canonical key for an execution result set, compared as an ORDER-INSENSITIVE
    MULTISET of rows with basic numeric normalization (1 == 1.0). Duplicate rows
    are preserved (multiset, not set). Empty result has its own dedicated key.
    """
    if not rows:
        return "EMPTY_RESULT"

    norm_rows = [tuple(_normalize_cell(c) for c in row) for row in rows]
    norm_rows.sort()  # order-insensitive; multiplicity preserved
    return str(norm_rows)


def agreement_bucket(agreement: float, majority_threshold: float = 0.5) -> str:
    """Bucket an agreement fraction: unanimous (all K) / strong-majority / split."""
    if agreement >= 1.0:
        return "unanimous"
    if agreement >= majority_threshold:
        return "strong-majority"
    return "split"


class ProgressTracker:
    """
    Thread-safe terminal progress tracker that prints completed count,
    percentage, throughput, and estimated time remaining (ETA) dynamically.
    """
    def __init__(self, total: int, initial: int = 0):
        self.total = total
        self.completed = initial
        self.start_time = time.time()
        self.lock = threading.Lock()

    def update(self) -> None:
        with self.lock:
            self.completed += 1
            # Adjust starting reference for throughput if we resumed
            elapsed = time.time() - self.start_time
            rate = self.completed / elapsed if elapsed > 0 else 0.0
            
            # Avoid division by zero when calculating ETA
            eta_seconds = (self.total - self.completed) / rate if rate > 0 else 0.0
            
            if eta_seconds > 3600:
                eta_str = f"{int(eta_seconds//3600)}h {int((eta_seconds%3600)//60)}m"
            elif eta_seconds > 60:
                eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s"
            else:
                eta_str = f"{int(eta_seconds)}s"

            percent = (self.completed / self.total) * 100
            
            # Progress bar visual representation (width: 30 chars)
            bar_width = 30
            filled_width = int(round(bar_width * self.completed / self.total)) if self.total > 0 else 0
            bar = "█" * filled_width + "-" * (bar_width - filled_width)
            
            # Write updating line to console
            sys.stdout.write(
                f"\rProgress: |{bar}| {self.completed}/{self.total} ({percent:.1f}%) "
                f"| Rate: {rate:.1f} rec/s | ETA: {eta_str}     "
            )
            sys.stdout.flush()


class BenchmarkPipeline:
    """
    Orchestrates candidate SQL generation, execution-based clustering,
    and evaluation against the ground-truth SQL query.
    Supports checkpointing to resume from interruptions.
    """
    def __init__(self, db_manager: DBManager, llm_client: BaseLLMClient):
        self.db_manager = db_manager
        self.llm_client = llm_client
        # Isolate checkpoints per output file so a smoke run and the full run
        # never share/clobber each other's partial state.
        results_stem = os.path.splitext(os.path.basename(settings.RESULTS_PATH))[0]
        self.checkpoint_dir = f"checkpoints_{results_stem}"

    def load_dataset(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Loads the dataset JSON from the specified path.
        """
        if not os.path.exists(data_path):
            logger.error(f"Dataset file not found at: {data_path}")
            raise FileNotFoundError(f"Dataset file not found at: {data_path}")
            
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Successfully loaded dataset from {data_path} containing {len(data)} records.")
            return data
        except Exception as e:
            logger.error(f"Error reading dataset file: {e}")
            raise

    def process_record(self, record: Dict[str, Any], record_idx: int) -> Dict[str, Any]:
        """
        Processes a single BIRD-SQL record:
        1. Retrieves DDL schema and executes the gold SQL.
        2. Generates K candidates, each with a distinct per-candidate seed.
        3. Executes candidates (caching identical SQL).
        4. Clusters SUCCESSFUL candidates by their normalized execution result.
        5. Derives agreement, the self-consistency prediction, in_pool and sc_correct.
        """
        rec_start = time.time()
        db_id = record.get("db_id") or record.get("dbId")
        question = record.get("question")
        evidence = record.get("evidence")
        ground_truth_sql = record.get("SQL") or record.get("sql") or record.get("ground_truth")
        question_id = record.get("question_id", record_idx)
        difficulty = record.get("difficulty")

        if not db_id or not question or not ground_truth_sql:
            logger.warning(f"Record {record_idx} is missing essential fields. Skipping.")
            return {
                "question_id": question_id,
                "error": "Missing essential fields",
                "success": False
            }

        logger.debug(f"Processing question ID {question_id} (DB: {db_id})")

        # 1. Schema + gold execution (the ground-truth result set)
        schema_ddl = self.db_manager.get_schema_ddl(db_id)
        question_with_hint = question
        if evidence and str(evidence).strip():
            question_with_hint = f"{question}\nHint: {evidence}"

        gt_query_res = self.db_manager.execute_query(db_id, ground_truth_sql)
        gold_success = gt_query_res.success
        gold_key = normalize_execution_result(gt_query_res.results) if gold_success else None

        # 2 & 3. Generate (per-candidate seed) and execute candidates
        K = settings.NUM_CANDIDATES
        candidates_meta: List[Dict[str, Any]] = []
        execution_cache: Dict[str, SQLQueryResult] = {}

        for i in range(K):
            try:
                cand_sql = self.llm_client.generate_sql(
                    schema_info=schema_ddl,
                    question=question_with_hint,
                    temperature=settings.GENERATION_TEMPERATURE,
                    seed=settings.GENERATION_SEED + i,
                )

                if cand_sql in execution_cache:
                    exec_res = execution_cache[cand_sql]
                else:
                    exec_res = self.db_manager.execute_query(db_id, cand_sql)
                    execution_cache[cand_sql] = exec_res

                if exec_res.success:
                    result_key = normalize_execution_result(exec_res.results)
                    matches_gold = bool(gold_success and result_key == gold_key)
                    error_msg = None
                else:
                    result_key = None  # failed candidates do NOT join a result cluster
                    matches_gold = False
                    error_msg = exec_res.error_message

                candidates_meta.append({
                    "candidate_index": i,
                    "sql": cand_sql,
                    "success": exec_res.success,
                    "execution_time_ms": exec_res.execution_time_ms,
                    "result_key": result_key,
                    "matches_gold": matches_gold,
                    "error": error_msg,
                })

            except Exception as e:
                logger.error(f"Error generating/evaluating candidate {i} for question {question_id}: {e}")
                candidates_meta.append({
                    "candidate_index": i,
                    "sql": "",
                    "success": False,
                    "execution_time_ms": 0.0,
                    "result_key": None,
                    "matches_gold": False,
                    "error": f"Generation error: {str(e)}",
                })

        # 4. Cluster ONLY successfully-executed candidates by normalized result.
        #    Failed candidates stay out of clusters but remain in the K denominator,
        #    so "all candidates crashed" reads as low agreement, not consensus.
        clusters_map: Dict[str, List[Dict[str, Any]]] = {}
        for cand in candidates_meta:
            if not cand["success"] or cand["result_key"] is None:
                continue
            clusters_map.setdefault(cand["result_key"], []).append(cand)

        # Deterministic order: largest first, ties broken by smallest candidate_index.
        sorted_clusters = sorted(
            clusters_map.values(),
            key=lambda cands: (-len(cands), min(c["candidate_index"] for c in cands)),
        )
        cluster_summaries = [
            {
                "result_key_preview": cands[0]["result_key"][:200],
                "size": len(cands),
                "candidate_indices": sorted(c["candidate_index"] for c in cands),
                "matches_gold": cands[0]["matches_gold"],
                "queries": sorted(set(c["sql"] for c in cands if c["sql"])),
            }
            for cands in sorted_clusters
        ]

        # 5. Self-consistency prediction = a candidate from the largest cluster.
        if sorted_clusters:
            champion = sorted_clusters[0]
            champion_size = len(champion)
            rep = min(champion, key=lambda c: c["candidate_index"])  # deterministic representative
            self_consistency_prediction = rep["sql"]
            champion_key_preview = rep["result_key"][:200]
            sc_correct = bool(rep["matches_gold"])
        else:
            champion_size = 0
            self_consistency_prediction = None
            champion_key_preview = None
            sc_correct = False

        agreement = champion_size / K if K > 0 else 0.0
        bucket = agreement_bucket(agreement, settings.MAJORITY_THRESHOLD)
        in_pool = any(c["matches_gold"] for c in candidates_meta)  # Pass@K oracle: ANY candidate matches gold
        challenger_size = len(sorted_clusters[1]) if len(sorted_clusters) > 1 else 0
        num_success = sum(1 for c in candidates_meta if c["success"])
        num_distinct_sql = len(set(c["sql"] for c in candidates_meta if c["sql"]))

        result = {
            "question_id": question_id,
            "db_id": db_id,
            "difficulty": difficulty,
            "question": question,
            "ground_truth_sql": ground_truth_sql,
            "gold_success": gold_success,
            "gold_result_preview": str(gt_query_res.results[:5]) if gold_success else None,
            "num_candidates": K,
            "num_successful_candidates": num_success,
            "num_distinct_candidate_sql": num_distinct_sql,
            "candidates": candidates_meta,
            "clusters": cluster_summaries,
            "champion_result_key_preview": champion_key_preview,
            "champion_size": champion_size,
            "challenger_size": challenger_size,
            "agreement": agreement,
            "agreement_bucket": bucket,
            "self_consistency_prediction": self_consistency_prediction,
            "in_pool": in_pool,
            "sc_correct": sc_correct,
            "processing_time_s": time.time() - rec_start,
            "success": True,
        }

        # Save checkpoint to disk
        self._write_checkpoint(question_id, result)

        return result

    def _write_checkpoint(self, question_id: int, result: Dict[str, Any]) -> None:
        """
        Safely write a single completed record checkpoint to disk.
        """
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(self.checkpoint_dir, f"{question_id}.json")
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to write checkpoint for question {question_id}: {e}")

    def _load_all_checkpoints(self) -> Dict[int, Dict[str, Any]]:
        """
        Load all successfully completed checkpoints from the checkpoint folder.
        """
        loaded = {}
        if os.path.exists(self.checkpoint_dir):
            for file_name in os.listdir(self.checkpoint_dir):
                if file_name.endswith(".json"):
                    ckpt_path = os.path.join(self.checkpoint_dir, file_name)
                    try:
                        with open(ckpt_path, "r", encoding="utf-8") as f:
                            ckpt_data = json.load(f)
                            q_id = ckpt_data.get("question_id")
                            if q_id is not None:
                                loaded[q_id] = ckpt_data
                    except Exception as e:
                        logger.warning(f"Corrupt checkpoint file skipped: {file_name}: {e}")
        return loaded

    def run(self) -> Dict[str, Any]:
        """
        Runs the benchmarking pipeline over the configured dataset.
        Saves the results to the results.json file.
        Resumes from existing checkpoints if available.
        """
        start_time = time.time()
        logger.info("Starting Text-to-SQL benchmarking pipeline...")

        # Load raw dataset
        raw_data = self.load_dataset(settings.DATA_PATH)

        # Optionally evaluate a seeded random sample (SUBSET_N) for smoke testing.
        # A seeded sample (rather than the first N) spreads the smoke run across
        # multiple databases and difficulties while staying reproducible.
        if settings.SUBSET_N is not None and settings.SUBSET_N < len(raw_data):
            rng = random.Random(settings.GENERATION_SEED)
            sample_idx = sorted(rng.sample(range(len(raw_data)), settings.SUBSET_N))
            dataset = [raw_data[i] for i in sample_idx]
            logger.info(
                f"SUBSET_N={settings.SUBSET_N}: evaluating a seeded sample of "
                f"{len(dataset)} of {len(raw_data)} records (seed={settings.GENERATION_SEED})."
            )
        else:
            dataset = raw_data
            logger.info(f"Evaluating all {len(dataset)} records.")

        total_records = len(dataset)
        processed_records: List[Dict[str, Any]] = []

        # Load checkpoints
        checkpoints = self._load_all_checkpoints()
        if checkpoints:
            logger.info(f"Found {len(checkpoints)} completed checkpoints. Resuming benchmark...")

        # Filter out records that are already in checkpoints
        to_process = []
        for idx, record in enumerate(dataset):
            q_id = record.get("question_id", idx)
            if q_id in checkpoints:
                processed_records.append(checkpoints[q_id])
            else:
                to_process.append((record, idx))

        logger.info(f"Records already completed: {len(processed_records)}. Records remaining to run: {len(to_process)}")

        # Initialize progress tracker
        progress_tracker = ProgressTracker(total=total_records, initial=len(processed_records))

        # Space before the bar; the bar itself draws on the first completed record.
        print("")

        if to_process:
            max_workers = settings.MAX_WORKERS
            logger.info(f"Running remaining executions with ThreadPoolExecutor (max_workers={max_workers})")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.process_record, record, idx): q_id
                    for record, idx in to_process
                    for q_id in [record.get("question_id", idx)]
                }

                for future in as_completed(futures):
                    q_id = futures[future]
                    try:
                        res = future.result()
                        processed_records.append(res)
                    except Exception as e:
                        logger.error(f"Exception raised processing record {q_id}: {e}")
                    finally:
                        # Update the terminal progress bar
                        progress_tracker.update()

        # Print a newline after progress bar completes
        print("\n")

        # Compute aggregate metrics. Only records whose GOLD executed are valid
        # for accuracy (a non-executing gold has no ground-truth result to match).
        valid_records = [
            r for r in processed_records if r.get("success") and r.get("gold_success")
        ]
        n = len(valid_records)
        if n == 0:
            logger.error("No records with a successfully-executing gold query.")
            return {"error": "No valid records"}

        sc_correct = sum(1 for r in valid_records if r.get("sc_correct"))
        in_pool = sum(1 for r in valid_records if r.get("in_pool"))
        sc_accuracy = sc_correct / n
        pass_at_k = in_pool / n
        gap = pass_at_k - sc_accuracy
        total_errors = n - sc_correct

        # Per-bucket: count, share, self-consistency accuracy, and share of all errors.
        bucket_names = ["unanimous", "strong-majority", "split"]
        bucket_metrics: Dict[str, Any] = {}
        for b in bucket_names:
            recs = [r for r in valid_records if r.get("agreement_bucket") == b]
            cnt = len(recs)
            correct = sum(1 for r in recs if r.get("sc_correct"))
            errors = cnt - correct
            bucket_metrics[b] = {
                "count": cnt,
                "share_of_total": cnt / n,
                "sc_accuracy": correct / cnt if cnt else 0.0,
                "errors": errors,
                "error_share": errors / total_errors if total_errors else 0.0,
            }

        avg_agreement = sum(r.get("agreement", 0.0) for r in valid_records) / n
        mean_distinct = sum(r.get("num_distinct_candidate_sql", 0) for r in valid_records) / n

        metrics = {
            "total_valid": n,
            "self_consistency_accuracy": sc_accuracy,
            "pass_at_k": pass_at_k,
            "generation_selection_gap": gap,
            "total_errors": total_errors,
            "average_agreement": avg_agreement,
            "mean_distinct_candidate_sql": mean_distinct,
            "buckets": bucket_metrics,
        }

        # Structure final audit file
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": time.time() - start_time,
            "settings": {
                "model": settings.LLM_MODEL,
                "num_candidates": settings.NUM_CANDIDATES,
                "temperature": settings.GENERATION_TEMPERATURE,
                "generation_seed": settings.GENERATION_SEED,
                "majority_threshold": settings.MAJORITY_THRESHOLD,
                "subset_n": settings.SUBSET_N,
                "max_workers": settings.MAX_WORKERS,
                "data_path": settings.DATA_PATH,
            },
            "metrics": metrics,
            "records": sorted(processed_records, key=lambda x: x.get("question_id", 0))
        }

        # Save to final results.json
        try:
            with open(settings.RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved evaluation results to {settings.RESULTS_PATH}")
            
            # Clean up temporary checkpoints upon successful completion of final output
            if os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
                logger.info("Cleaned up temporary checkpoint files.")
                
        except Exception as e:
            logger.error(f"Failed to persist results or delete checkpoints: {e}")

        logger.info(
            f"Pipeline Complete. Valid: {n}. "
            f"Self-consistency accuracy: {sc_accuracy:.4f}. "
            f"Pass@K: {pass_at_k:.4f}. Gap: {gap:.4f}. "
            f"Avg agreement: {avg_agreement:.3f}. Mean distinct SQL/q: {mean_distinct:.2f}."
        )

        return output_data
