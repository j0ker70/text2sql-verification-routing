import re
import time
import logging
import requests
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from config import settings

logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    """
    Abstract Base Class for LLM Client.
    """
    @abstractmethod
    def generate_sql(
        self,
        schema_info: str,
        question: str,
        temperature: float = 0.7,
        seed: Optional[int] = None
    ) -> str:
        """
        Generates a SQL query given a schema and a natural language question.

        Args:
            schema_info: Text representation of database schema.
            question: Natural language question.
            temperature: LLM temperature setting.
            seed: Optional RNG seed for reproducible sampling. Use a DIFFERENT
                seed per candidate to get reproducible-yet-diverse samples.

        Returns:
            The raw generated SQL query as a string.
        """
        pass


class OllamaClient(BaseLLMClient):
    """
    Concrete implementation of BaseLLMClient for Ollama API.
    """
    def __init__(
        self, 
        base_url: str = settings.OLLAMA_BASE_URL, 
        model: str = settings.LLM_MODEL,
        timeout: float = settings.LLM_TIMEOUT,
        max_retries: int = settings.LLM_MAX_RETRIES
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def _clean_sql(self, raw_output: str) -> str:
        """
        Strips markdown wrappers, quotes, and extra whitespace to ensure
        the output is a valid, raw SQL query.
        """
        if not raw_output:
            return ""
        
        # Strip code blocks like ```sql ... ``` or ``` ... ```
        pattern = r"```(?:sql)?\s*(.*?)\s*```"
        match = re.search(pattern, raw_output, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1)
        else:
            sql = raw_output
            
        sql = sql.strip()
        
        # Remove any leading/trailing quotes that model might have added
        if sql.startswith('"') and sql.endswith('"'):
            sql = sql[1:-1].strip()
        elif sql.startswith("'") and sql.endswith("'"):
            sql = sql[1:-1].strip()
            
        # Strip trailing semicolon if present (standardizing queries)
        if sql.endswith(";"):
            sql = sql[:-1].strip()
            
        return sql

    def generate_sql(
        self,
        schema_info: str,
        question: str,
        temperature: float = settings.GENERATION_TEMPERATURE,
        seed: Optional[int] = None
    ) -> str:
        """
        Call Ollama chat endpoint to translate a schema and question to raw SQL.
        """
        url = f"{self.base_url}/api/chat"
        
        system_instructions = (
            "You are a precise Text-to-SQL translation engine. "
            "Your task is to convert the database schema and question into a valid SQLite query. "
            "You must output ONLY the raw SQL query. Do not wrap your response in markdown code blocks, "
            "do not include the word 'sql', do not provide explanations, and do not prefix/suffix "
            "the SQL query with any text."
        )
        
        prompt = (
            f"SQLite Database Schema:\n{schema_info}\n\n"
            f"Question:\n{question}\n\n"
            f"Generate SQLite query:"
        )
        
        options: Dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        if settings.GENERATION_NUM_PREDICT:
            options["num_predict"] = settings.GENERATION_NUM_PREDICT

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": options
        }
        
        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    f"Sending request to Ollama (attempt {attempt}/{self.max_retries}) "
                    f"for model '{self.model}'"
                )
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                
                response_json = response.json()
                raw_content = response_json["message"]["content"]
                cleaned_sql = self._clean_sql(raw_content)
                return cleaned_sql
                
            except requests.RequestException as e:
                last_exception = e
                logger.warning(
                    f"Ollama request failed on attempt {attempt}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries:
                    # Exponential backoff: 2s, 4s, 8s
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
            except (KeyError, ValueError) as e:
                last_exception = e
                logger.warning(
                    f"Invalid response format from Ollama on attempt {attempt}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        # If we got here, all attempts failed
        error_msg = f"Failed to generate SQL after {self.max_retries} attempts: {last_exception}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
