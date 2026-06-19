import os
import sqlite3
import time
import logging
from typing import Any, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SQLQueryResult:
    """
    Standardized container for SQL execution results.
    """
    success: bool
    results: List[Tuple[Any, ...]]
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


class SQLiteProgressTimeout:
    """
    A callback helper for sqlite3.set_progress_handler to enforce execution timeouts.
    """
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()

    def __call__(self) -> int:
        if time.time() - self.start_time > self.timeout_seconds:
            # Returning a non-zero value aborts the query and raises sqlite3.OperationalError
            return 1
        return 0


class DBManager:
    """
    Handles safe execution of SQL queries against sqlite databases.
    """
    def __init__(self, db_root_path: str):
        self.db_root_path = db_root_path

    def get_db_path(self, db_id: str) -> str:
        """
        Dynamically construct the path to the sqlite database for the given db_id.
        """
        return os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")

    def get_schema_ddl(self, db_id: str) -> str:
        """
        Extract CREATE TABLE statements for the given db_id database.
        """
        db_path = self.get_db_path(db_id)
        if not os.path.exists(db_path):
            logger.error(f"Database file not found at: {db_path}")
            return ""
        
        conn = None
        try:
            db_uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(db_uri, uri=True)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            rows = cursor.fetchall()
            ddls = [row[0] for row in rows if row[0]]
            return "\n\n".join(ddls)
        except Exception as e:
            logger.error(f"Failed to retrieve schema DDL for DB '{db_id}': {e}")
            return ""
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass


    def execute_query(
        self, 
        db_id: str, 
        sql: str, 
        timeout_seconds: float = 10.0
    ) -> SQLQueryResult:
        """
        Execute a query against a BIRD SQLite database.
        
        Args:
            db_id: The ID of the database (matches folder name).
            sql: The SQL query string to run.
            timeout_seconds: Maximum execution duration before raising a timeout error.

        Returns:
            SQLQueryResult object.
        """
        db_path = self.get_db_path(db_id)
        
        if not os.path.exists(db_path):
            error_msg = f"Database file not found at: {db_path}"
            logger.error(error_msg)
            return SQLQueryResult(success=False, results=[], error_message=error_msg)

        conn = None
        start_time = time.time()
        try:
            # Connect to database in read-only mode for safety
            db_uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(db_uri, uri=True)
            cursor = conn.cursor()

            # Set progress handler to check for query timeouts every 100 SQLite instructions
            timeout_handler = SQLiteProgressTimeout(timeout_seconds)
            conn.set_progress_handler(timeout_handler, 100)

            cursor.execute(sql)
            rows = cursor.fetchall()
            
            execution_time_ms = (time.time() - start_time) * 1000.0
            return SQLQueryResult(
                success=True, 
                results=rows, 
                execution_time_ms=execution_time_ms
            )

        except sqlite3.OperationalError as e:
            execution_time_ms = (time.time() - start_time) * 1000.0
            error_msg = str(e)
            if "interrupted" in error_msg.lower() or (time.time() - start_time) >= timeout_seconds:
                error_msg = f"Query timed out after {timeout_seconds}s: {error_msg}"
                logger.warning(f"Timeout executing SQL on DB '{db_id}': {sql[:200]}...")
            else:
                logger.warning(f"Operational error on DB '{db_id}': {e}. SQL: {sql[:200]}...")
            
            return SQLQueryResult(
                success=False, 
                results=[], 
                error_message=error_msg, 
                execution_time_ms=execution_time_ms
            )
            
        except sqlite3.DatabaseError as e:
            execution_time_ms = (time.time() - start_time) * 1000.0
            logger.warning(f"Database error on DB '{db_id}': {e}. SQL: {sql[:200]}...")
            return SQLQueryResult(
                success=False, 
                results=[], 
                error_message=str(e), 
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000.0
            logger.error(f"Unexpected error executing SQL on DB '{db_id}': {e}. SQL: {sql[:200]}...")
            return SQLQueryResult(
                success=False, 
                results=[], 
                error_message=str(e), 
                execution_time_ms=execution_time_ms
            )
            
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug(f"Failed to close connection: {e}")
