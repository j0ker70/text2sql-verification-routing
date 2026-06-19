import os
import json
import sqlite3
import logging
import requests
import sys
from typing import Dict, Any

from config import settings, setup_logging
from db_manager import DBManager
from llm_client import OllamaClient
from pipeline import BenchmarkPipeline

logger = logging.getLogger(__name__)

def ensure_mock_data_if_missing() -> None:
    """
    Check if the dataset and database directory exist. If they do not, 
    generate a mock SQLite database and a mock dev.json file so that the
    pipeline runs out-of-the-box.
    """
    db_missing = not os.path.exists(settings.DB_ROOT_PATH) or not os.listdir(settings.DB_ROOT_PATH) if os.path.exists(settings.DB_ROOT_PATH) else True
    data_missing = not os.path.exists(settings.DATA_PATH)
    
    if db_missing or data_missing:
        logger.info(
            "Configured dataset or SQLite databases not found. "
            "Generating a mock dataset and SQLite database for pipeline demonstration..."
        )
        
        # 1. Create Mock Database Directory & File
        db_id = "mock_school"
        db_dir = os.path.join(settings.DB_ROOT_PATH, db_id)
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, f"{db_id}.sqlite")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Setup tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    grade INTEGER,
                    enrollment_year INTEGER
                );
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classes (
                    class_id INTEGER PRIMARY KEY,
                    class_name TEXT NOT NULL,
                    teacher_name TEXT NOT NULL
                );
            """)
            
            # Clean and Insert Data
            cursor.execute("DELETE FROM students;")
            cursor.execute("DELETE FROM classes;")
            
            cursor.executemany("""
                INSERT INTO students (student_id, name, grade, enrollment_year)
                VALUES (?, ?, ?, ?);
            """, [
                (1, "Alice Smith", 12, 2024),
                (2, "Bob Jones", 11, 2025),
                (3, "Charlie Brown", 12, 2024),
                (4, "Diana Prince", 10, 2026),
                (5, "Evan Wright", 11, 2025)
            ])
            
            cursor.executemany("""
                INSERT INTO classes (class_id, class_name, teacher_name)
                VALUES (?, ?, ?);
            """, [
                (101, "Advanced Calculus", "Mr. Gauss"),
                (102, "Introductory Physics", "Dr. Newton"),
                (103, "World History", "Mrs. Herodotus")
            ])
            
            conn.commit()
            conn.close()
            logger.info(f"Created mock SQLite database at {db_path}")
        except Exception as e:
            logger.error(f"Failed to generate mock database: {e}")
            raise

        # 2. Create Mock dev.json File
        mock_dev = [
            {
                "question_id": 1,
                "db_id": "mock_school",
                "question": "What are the names of students in grade 12?",
                "evidence": "Filter students where grade is equal to 12.",
                "SQL": "SELECT name FROM students WHERE grade = 12;"
            },
            {
                "question_id": 2,
                "db_id": "mock_school",
                "question": "How many students are enrolled in total?",
                "evidence": "Count all records in the students table.",
                "SQL": "SELECT COUNT(*) FROM students;"
            },
            {
                "question_id": 3,
                "db_id": "mock_school",
                "question": "List the names of all teachers and the classes they teach.",
                "evidence": "Show classes table attributes: teacher_name and class_name.",
                "SQL": "SELECT teacher_name, class_name FROM classes;"
            },
            {
                "question_id": 4,
                "db_id": "mock_school",
                "question": "Get the enrollment year for Bob Jones.",
                "evidence": "Match student name to 'Bob Jones'.",
                "SQL": "SELECT enrollment_year FROM students WHERE name = 'Bob Jones';"
            }
        ]
        
        try:
            with open(settings.DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(mock_dev, f, indent=2, ensure_ascii=False)
            logger.info(f"Created mock dataset file at {settings.DATA_PATH}")
        except Exception as e:
            logger.error(f"Failed to write mock dataset: {e}")
            raise


def check_ollama_status() -> bool:
    """
    Checks if the Ollama server is reachable.
    """
    try:
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/", timeout=2.0)
        if response.status_code == 200:
            logger.info(f"Connected to Ollama server at {settings.OLLAMA_BASE_URL}")
            return True
    except requests.RequestException:
        pass
    
    logger.warning(
        f"Unable to connect to Ollama server at: {settings.OLLAMA_BASE_URL}\n"
        f"Please verify:\n"
        f"  1. Ollama is running (`ollama serve` or open the desktop app)\n"
        f"  2. You have pulled the configured model: `ollama pull {settings.LLM_MODEL}`\n"
    )
    return False


def main() -> None:
    # 1. Setup logging
    setup_logging()
    logger.info("Initializing benchmarking pipeline...")
    
    # 2. Check and generate mock data if needed
    ensure_mock_data_if_missing()
    
    # 3. Check LLM client server status
    check_ollama_status()
    
    # 4. Instantiate clients and managers
    db_manager = DBManager(db_root_path=settings.DB_ROOT_PATH)
    llm_client = OllamaClient(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.LLM_MODEL,
        timeout=settings.LLM_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES
    )
    
    # 5. Run the pipeline
    pipeline = BenchmarkPipeline(db_manager=db_manager, llm_client=llm_client)
    try:
        output_data = pipeline.run()
        
        # Display short overview of the run
        if "metrics" in output_data:
            m = output_data["metrics"]
            print("\n" + "="*60)
            print("SELF-CONSISTENCY BENCHMARK SUMMARY")
            print("="*60)
            print(f"Valid questions:             {m['total_valid']}")
            print(f"Self-consistency accuracy:   {m['self_consistency_accuracy']:.2%}")
            print(f"Pass@K (oracle upper bound): {m['pass_at_k']:.2%}")
            print(f"Generation-selection gap:    {m['generation_selection_gap']:.2%}")
            print(f"Average agreement:           {m['average_agreement']:.2%}")
            print(f"Mean distinct SQL / question:{m['mean_distinct_candidate_sql']:.2f}")
            print("-"*60)
            print(f"{'bucket':<16}{'n':>5}{'share':>9}{'sc_acc':>9}{'err_share':>11}")
            for b in ["unanimous", "strong-majority", "split"]:
                bm = m["buckets"][b]
                print(f"{b:<16}{bm['count']:>5}{bm['share_of_total']:>8.0%}{bm['sc_accuracy']:>9.1%}{bm['error_share']:>11.1%}")
            print("="*60)
            print(f"Detailed logs saved to: {settings.RESULTS_PATH}")
            print("="*60 + "\n")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
