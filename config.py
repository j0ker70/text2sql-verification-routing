import os
import logging
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or a .env file.
    Provides sane defaults for out-of-the-box usage.
    """
    # LLM Settings
    LLM_MODEL: str = "qwen2.5-coder:7b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GENERATION_TEMPERATURE: float = 0.7
    GENERATION_SEED: int = 42          # base seed; candidate i is sampled with seed = GENERATION_SEED + i
    GENERATION_NUM_PREDICT: int = 512  # cap on generated tokens per candidate
    LLM_TIMEOUT: float = 60.0  # timeout in seconds for LLM calls
    LLM_MAX_RETRIES: int = 3   # number of retries for LLM requests

    # Paths and Dataset Settings
    DATA_PATH: str = "mini_dev_sqlite.json"  # official BIRD Mini-Dev 500 (SQLite)
    DB_ROOT_PATH: str = "databases"
    SUBSET_N: Optional[int] = None     # if set, evaluate a seeded random sample of N records (smoke testing)
    NUM_CANDIDATES: int = 8            # K candidates generated per question
    MAJORITY_THRESHOLD: float = 0.5    # agreement in [MAJORITY_THRESHOLD, 1.0) => strong-majority; below => split
    MAX_WORKERS: int = 4

    # Logging & Run Settings
    LOG_LEVEL: str = "INFO"
    RESULTS_PATH: str = "results.json"
    LOG_TO_FILE: bool = True
    LOG_FILE_PATH: str = "benchmark.log"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Setup logging configuration globally
def setup_logging() -> None:
    numeric_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    handlers: list = []
    if settings.LOG_TO_FILE:
        handlers.append(logging.FileHandler(settings.LOG_FILE_PATH, encoding="utf-8"))
    else:
        handlers.append(logging.StreamHandler())
        
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )

