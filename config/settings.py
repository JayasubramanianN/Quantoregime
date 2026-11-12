import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "QuantMacro AI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    
    # Base directories
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    
    # API Keys & Credentials
    FRED_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    SEC_USER_AGENT: str = "QuantMacroAI Analyst analyst@quantmacro.org"
    
    # Vector store parameters
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_DIMENSION: int = 384
    INDEX_TYPE: str = "faiss"
    
    # Statistical / Hypothesis testing defaults
    SIGNIFICANCE_LEVEL: float = 0.05
    MAX_GRANGER_LAGS: int = 5
    
    # Monte Carlo simulation defaults
    DEFAULT_MC_PATHS: int = 10000
    DEFAULT_FORECAST_DAYS: int = 252
    RANDOM_SEED: int = 42

settings = Settings()

# Ensure directories exist
settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
