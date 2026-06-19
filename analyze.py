"""
Phase 3 analysis / reporting for the disagreement-vs-error probe.

Reads a results JSON produced by the pipeline (default: results.json) and reports:
  1. Overall self-consistency accuracy.
  2. Pass@K oracle upper bound (in_pool rate).
  3. Generation-selection gap (Pass@K - self-consistency accuracy).
  4. Agreement buckets (unanimous / strong-majority / split): count, SC accuracy,
     and share of total errors.
  5. Routing curve: sort queries by ascending agreement and, as the routing budget
     grows 0% -> 100% (most-disagreeing first), tabulate the fraction of
     self-consistency ERRORS that budget covers.

Plus smoke-time diagnostics: candidate diversity (distinct SQL per question) and a
throughput-based timing estimate. Pure analysis over saved raw results - no
regeneration, so it is cheap to re-run.

Usage:
    python analyze.py [results.json] [--show-candidates N] [--full-target 500]
"""
import sys
import json
import csv
import argparse
from collections import Counter
from typing import Any, Dict, List

BUCKETS = ["unanimous", "strong-majority", "split"]


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def valid_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Records that were processed AND whose gold executed (eligible for accuracy)."""
    return [r for r in data.get("records", []) if r.get("success") and r.get("gold_success")]


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


def avg(xs: List[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def crash_rate(r: Dict[str, Any]) -> float:
    """Fraction of a query's K candidates that FAILED to execute."""
    K = r.get("num_candidates", 0) or 0
    ns = r.get("num_successful_candidates", 0)
    return (K - ns) / K if K else 0.0


def exec_agreement(r: Dict[str, Any]):
    """
    agreement_exec = largest execution-result cluster / number of candidates that
    executed successfully. Defined only for queries with >= 2 successful candidates
    (otherwise there is no result-disagreement to measure). Returns None if undefined.
    """
    ns = r.get("num_successful_candidates", 0)
    if ns >= 2:
        return r.get("champion_size", 0) / ns
    return None


def bucket_of(value: float, majority: float = 0.5) -> str:
    if value >= 1.0:
        return "unanimous"
    if value >= majority:
        return "strong-majority"
    return "split"


def routing_curve_keyed(records: List[Dict[str, Any]], sort_key) -> List[Dict[str, Any]]:
    """
    Order records by sort_key (lowest first = routed first), and for each prefix
    (budget) report the cumulative fraction of self-consistency errors covered.
    """
    ordered = sorted(records, key=sort_key)
    n = len(ordered)
    total_errors = sum(1 for r in ordered if not r.get("sc_correct"))
    rows = []
    cum_err = 0
    for i, r in enumerate(ordered, start=1):
        if not r.get("sc_correct"):
            cum_err += 1
        rows.append({
            "queries_sent": i,
            "budget_frac": i / n if n else 0.0,
            "cum_errors_covered": cum_err,
            "frac_errors_covered": (cum_err / total_errors) if total_errors else 0.0,
        })
    return rows


def routing_curve(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Original routing curve: most-disagreeing first by agreement = largest cluster / K.
    Deterministic tie-break: lower agreement, then question_id.
    """
    return routing_curve_keyed(records, lambda r: (r.get("agreement", 0.0), r.get("question_id", 0)))


def auc(curve: List[Dict[str, Any]]) -> float:
    """Area under the (budget, error-coverage) curve. Random ~= 0.5; concentrated -> 1.0."""
    if not curve:
        return 0.0
    x = [0.0] + [p["budget_frac"] for p in curve]
    y = [0.0] + [p["frac_errors_covered"] for p in curve]
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return area


def separated_analysis(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Isolate genuine result-disagreement from crash-driven disagreement.

      - agreement_exec = largest exec-result cluster / #successful candidates,
        over the executable-only subset (>= 2 successful candidates).
      - crash rate analysis: does the share of crashing candidates predict errors?
      - confident-but-wrong ceiling: error rate of the unanimous bucket(s).

    The self-consistency PREDICTION (largest exec cluster) is unchanged; only the
    agreement *measure* and the subset change, so sc_correct is reused as-is.
    """
    n = len(recs)
    K = recs[0].get("num_candidates", 0) if recs else 0
    total_err = sum(1 for r in recs if not r.get("sc_correct"))

    print("\n" + "=" * 64)
    print("SEPARATED ANALYSIS  (result-disagreement vs crashes)")
    print("=" * 64)

    ns_dist = Counter(r.get("num_successful_candidates", 0) for r in recs)
    print(f"K={K}; #successful-candidates distribution (ns:#queries): "
          + ", ".join(f"{k}:{ns_dist[k]}" for k in sorted(ns_dist)))

    exe = [r for r in recs if r.get("num_successful_candidates", 0) >= 2]
    n_exe = len(exe)
    n_allcrash = sum(1 for r in recs if r.get("num_successful_candidates", 0) == 0)
    n_single = sum(1 for r in recs if r.get("num_successful_candidates", 0) == 1)
    err_allcrash = sum(1 for r in recs if r.get("num_successful_candidates", 0) == 0 and not r.get("sc_correct"))
    err_single = sum(1 for r in recs if r.get("num_successful_candidates", 0) == 1 and not r.get("sc_correct"))
    for r in exe:
        r["_agr_exec"] = r["champion_size"] / r["num_successful_candidates"]
    exe_err = sum(1 for r in exe if not r.get("sc_correct"))
    print(f"\nExecutable-only subset (>=2 successful): {n_exe} of {n}  "
          f"[excluded {n_allcrash} all-crash ({err_allcrash} err), {n_single} single-success ({err_single} err)]")
    print(f"of {total_err} total SC-errors: {exe_err} are in the executable subset, "
          f"{err_allcrash + err_single} are crash-dominated (excluded).")

    # (1+2) agreement_exec buckets + routing
    print("\n-- (1) buckets by agreement_exec  (largest exec cluster / #successful) --")
    print(f"{'bucket':<16}{'n':>5}{'share':>9}{'sc_acc':>9}{'errors':>8}{'err_share':>11}")
    bucket_out = {}
    for b in BUCKETS:
        br = [r for r in exe if bucket_of(r["_agr_exec"]) == b]
        cnt = len(br)
        correct = sum(1 for r in br if r.get("sc_correct"))
        err = cnt - correct
        acc = correct / cnt if cnt else 0.0
        es = err / exe_err if exe_err else 0.0
        bucket_out[b] = {"count": cnt, "sc_accuracy": acc, "error_share": es}
        share = pct(cnt / n_exe) if n_exe else "n/a"
        print(f"{b:<16}{cnt:>5}{share:>9}{pct(acc):>9}{err:>8}{pct(es):>11}")

    exec_auc = None
    print("\n-- (2) routing by agreement_exec (most result-disagreement first) --")
    if exe_err and n_exe:
        curve = routing_curve_keyed(exe, lambda r: (r["_agr_exec"], r.get("question_id", 0)))
        for mark in (0.10, 0.20, 0.30, 0.50, 0.75, 1.00):
            pts = [p for p in curve if p["budget_frac"] <= mark + 1e-9]
            fe = pts[-1]["frac_errors_covered"] if pts else 0.0
            ec = pts[-1]["cum_errors_covered"] if pts else 0
            print(f"  budget {pct(mark):>6}: {pct(fe):>6} of errors ({ec}/{exe_err})")
        exec_auc = auc(curve)
        verdict = "STILL predicts" if exec_auc > 0.55 else ("weakly predicts" if exec_auc > 0.5 else "does NOT predict")
        print(f"  AUC={exec_auc:.3f} (random 0.500)  ->  result-disagreement {verdict} SC errors once crashes are removed")
    else:
        print("  (no errors in executable subset; routing undefined)")

    # (3) crash-rate analysis
    print("\n-- (3) does crash rate alone predict SC errors? --")
    mcr_ok = avg([crash_rate(r) for r in recs if r.get("sc_correct")])
    mcr_err = avg([crash_rate(r) for r in recs if not r.get("sc_correct")])
    print(f"mean crash rate | SC-correct: {pct(mcr_ok)}   SC-error: {pct(mcr_err)}")

    def crash_band(r):
        c = (r.get("num_candidates", 0) or 0) - r.get("num_successful_candidates", 0)
        if c == 0:
            return "0 crashes"
        if c <= 3:
            return "1-3"
        if c < (r.get("num_candidates", 0) or 0):
            return "4-7"
        return "all (8)"

    print(f"{'crash band':<12}{'n':>5}{'sc_acc':>9}{'errors':>8}{'err_share':>11}")
    for b in ["0 crashes", "1-3", "4-7", "all (8)"]:
        br = [r for r in recs if crash_band(r) == b]
        cnt = len(br)
        if not cnt:
            continue
        correct = sum(1 for r in br if r.get("sc_correct"))
        err = cnt - correct
        es = pct(err / total_err) if total_err else "n/a"
        print(f"{b:<12}{cnt:>5}{pct(correct/cnt):>9}{err:>8}{es:>11}")

    crash_auc = None
    if total_err:
        ccurve = routing_curve_keyed(recs, lambda r: (-crash_rate(r), r.get("question_id", 0)))
        crash_auc = auc(ccurve)
        verdict = "predicts" if crash_auc > 0.55 else ("weakly predicts" if crash_auc > 0.5 else "does NOT predict")
        print(f"routing by crash-rate (most-crashing first): AUC={crash_auc:.3f} (random 0.500)  ->  crash rate alone {verdict} SC errors")

    # (4) confident-but-wrong ceiling
    print("\n-- (4) CONFIDENT-BUT-WRONG CEILING (unreachable by disagreement routing) --")
    unum = [r for r in recs if r.get("agreement", 0.0) >= 1.0]   # all K agree
    u_err = sum(1 for r in unum if not r.get("sc_correct"))
    u_rate = u_err / len(unum) if unum else 0.0
    print(f"original unanimous (all {K} candidates agree): n={len(unum)}  errors={u_err}  "
          f"error rate={pct(u_rate)}")
    eu = [r for r in exe if r["_agr_exec"] >= 1.0]
    eu_err = sum(1 for r in eu if not r.get("sc_correct"))
    eu_rate = eu_err / len(eu) if eu else 0.0
    print(f"exec-unanimous (all SUCCESSFUL candidates agree, ns>=2): n={len(eu)}  errors={eu_err}  "
          f"error rate={pct(eu_rate)}")
    print(f"=> even a perfect disagreement router leaves ~{pct(u_rate)} of unanimous queries wrong "
          f"(these are confident-but-wrong and cannot be flagged by disagreement).")

    return {
        "n_valid": n,
        "n_executable_subset": n_exe,
        "n_all_crash": n_allcrash,
        "n_single_success": n_single,
        "errors_total": total_err,
        "errors_in_executable_subset": exe_err,
        "errors_crash_dominated": err_allcrash + err_single,
        "agreement_exec_buckets": bucket_out,
        "agreement_exec_routing_auc": exec_auc,
        "crash_routing_auc": crash_auc,
        "mean_crash_rate_correct": mcr_ok,
        "mean_crash_rate_error": mcr_err,
        "unanimous_error_rate_confident_but_wrong": u_rate,
        "exec_unanimous_error_rate": eu_rate,
    }


def report(path: str, show_candidates: int, full_target: int) -> None:
    data = load(path)
    recs = valid_records(data)
    n = len(recs)
    s = data.get("settings", {})
    m = data.get("metrics", {})

    print("=" * 64)
    print(f"ANALYSIS OF: {path}")
    print("=" * 64)
    print(f"model={s.get('model')}  K={s.get('num_candidates')}  temp={s.get('temperature')}  "
          f"seed={s.get('generation_seed')}  subset_n={s.get('subset_n')}")
    print(f"valid questions (gold executed): {n}")
    if n == 0:
        print("No valid records to analyze.")
        return

    sc_correct = sum(1 for r in recs if r.get("sc_correct"))
    in_pool = sum(1 for r in recs if r.get("in_pool"))
    sc_acc = sc_correct / n
    passk = in_pool / n
    gap = passk - sc_acc
    total_errors = n - sc_correct

    # ---- 1-3. Headline correctness -----------------------------------------
    print("\n-- CORRECTNESS --------------------------------------------------")
    print(f"1. Self-consistency accuracy : {pct(sc_acc)}  ({sc_correct}/{n})")
    print(f"2. Pass@K oracle upper bound : {pct(passk)}  ({in_pool}/{n})")
    print(f"3. Generation-selection gap  : {pct(gap)}  (selection leaves this much on the table)")

    # ---- Diversity diagnostic (key smoke check) ----------------------------
    distinct_hist = Counter(r.get("num_distinct_candidate_sql", 0) for r in recs)
    mean_distinct = sum(r.get("num_distinct_candidate_sql", 0) for r in recs) / n
    all_identical = sum(1 for r in recs if r.get("num_distinct_candidate_sql", 0) <= 1)
    K = s.get("num_candidates", 0)
    print("\n-- CANDIDATE DIVERSITY ------------------------------------------")
    print(f"mean distinct SQL / question : {mean_distinct:.2f} of K={K}")
    print(f"queries with all-identical candidates: {all_identical}/{n} ({pct(all_identical/n)})")
    print("distinct-SQL histogram (distinct_count: #queries): "
          + ", ".join(f"{k}:{distinct_hist[k]}" for k in sorted(distinct_hist)))

    # ---- 4. Agreement buckets ----------------------------------------------
    print("\n-- AGREEMENT DISTRIBUTION & BUCKETS ------------------------------")
    print(f"{'bucket':<16}{'n':>5}{'share':>9}{'sc_acc':>9}{'errors':>8}{'err_share':>11}")
    for b in BUCKETS:
        br = [r for r in recs if r.get("agreement_bucket") == b]
        cnt = len(br)
        correct = sum(1 for r in br if r.get("sc_correct"))
        errors = cnt - correct
        acc = correct / cnt if cnt else 0.0
        eshare = errors / total_errors if total_errors else 0.0
        print(f"{b:<16}{cnt:>5}{pct(cnt/n):>9}{pct(acc):>9}{errors:>8}{pct(eshare):>11}")
    # raw agreement histogram (fractions i/K)
    agr_hist = Counter(round(r.get("agreement", 0.0), 3) for r in recs)
    print("agreement histogram (value: #queries): "
          + ", ".join(f"{a:g}:{agr_hist[a]}" for a in sorted(agr_hist)))

    # ---- 5. Routing curve --------------------------------------------------
    curve = routing_curve(recs)
    print("\n-- ROUTING CURVE (most-disagreeing first) -----------------------")
    if total_errors == 0:
        print("No self-consistency errors in this set - routing curve is undefined.")
    else:
        print(f"total errors to cover: {total_errors}")
        print(f"{'budget':>8}{'queries':>9}{'errors_covered':>16}{'frac_errors':>13}")
        for mark in (0.10, 0.20, 0.30, 0.50, 0.75, 1.00):
            # curve is ascending in budget_frac; take the last point at/below the mark
            pts = [p for p in curve if p["budget_frac"] <= mark + 1e-9]
            if pts:
                pt = pts[-1]
                qs, ec, fe = pt["queries_sent"], pt["cum_errors_covered"], pt["frac_errors_covered"]
            else:
                qs, ec, fe = 0, 0, 0.0  # budget too small to send even one query
            print(f"{pct(mark):>8}{qs:>9}{ec:>16}{pct(fe):>13}")
        # budget needed to cover X% of errors
        def budget_for(frac: float) -> str:
            for p in curve:
                if p["frac_errors_covered"] >= frac - 1e-9:
                    return pct(p["budget_frac"])
            return "100.0%"
        print(f"budget to cover 50% of errors: {budget_for(0.50)}; "
              f"80%: {budget_for(0.80)}; 100%: {budget_for(1.00)}")
        print(f"AUC (error-coverage vs budget): {auc(curve):.3f}  (random baseline 0.500; higher = "
              f"errors concentrated at low agreement)")

    # ---- Timing & extrapolation --------------------------------------------
    elapsed = data.get("elapsed_seconds", 0.0)
    per_rec_times = [r.get("processing_time_s", 0.0) for r in recs]
    mean_rec = sum(per_rec_times) / n if n else 0.0
    throughput = n / elapsed if elapsed > 0 else 0.0
    print("\n-- TIMING -------------------------------------------------------")
    print(f"wall-clock elapsed           : {elapsed:.1f}s for {n} questions")
    print(f"mean per-question wall time  : {mean_rec:.1f}s (sum of K gens+execs, no concurrency credit)")
    print(f"effective throughput         : {throughput:.3f} questions/s at max_workers={s.get('max_workers')}")
    if throughput > 0:
        est = full_target / throughput
        h, rem = divmod(int(est), 3600)
        mnt = rem // 60
        print(f"=> estimated time for {full_target} questions: {est:.0f}s (~{h}h {mnt}m) at the same settings")

    # ---- Optional: print candidate SQLs for the most-disagreeing queries ----
    if show_candidates > 0:
        ordered = sorted(recs, key=lambda r: (r.get("agreement", 0.0), r.get("question_id", 0)))
        print("\n-- CANDIDATE SQL INSPECTION (lowest-agreement first) ------------")
        for r in ordered[:show_candidates]:
            print(f"\nQ{r.get('question_id')} [{r.get('db_id')}] agreement={r.get('agreement'):.3f} "
                  f"bucket={r.get('agreement_bucket')} sc_correct={r.get('sc_correct')} in_pool={r.get('in_pool')}")
            print(f"  Q: {r.get('question')[:140]}")
            for c in r.get("candidates", []):
                tag = "ok " if c.get("success") else "ERR"
                gold = "*" if c.get("matches_gold") else " "
                sql = (c.get("sql") or c.get("error") or "")[:130]
                print(f"   [{c.get('candidate_index')}]{gold}{tag} {sql}")

    # ---- Separated analysis: result-disagreement vs crashes ----------------
    separated = separated_analysis(recs)

    # ---- Save analysis + routing CSV ---------------------------------------
    out = {
        "source": path,
        "n_valid": n,
        "self_consistency_accuracy": sc_acc,
        "pass_at_k": passk,
        "generation_selection_gap": gap,
        "total_errors": total_errors,
        "mean_distinct_candidate_sql": mean_distinct,
        "buckets": m.get("buckets", {}),
        "routing_auc": auc(curve) if total_errors else None,
        "separated": separated,
    }
    with open("analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open("routing_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["queries_sent", "budget_frac", "cum_errors_covered", "frac_errors_covered"])
        w.writeheader()
        w.writerows(curve)
    print("\nSaved analysis.json and routing_curve.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results.json")
    ap.add_argument("--show-candidates", type=int, default=0,
                    help="print K candidate SQLs for the N lowest-agreement queries")
    ap.add_argument("--full-target", type=int, default=500,
                    help="question count to extrapolate the timing estimate to")
    args = ap.parse_args()
    report(args.results, args.show_candidates, args.full_target)
