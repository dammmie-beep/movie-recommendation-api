"""
Configuration settings for the application.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_PATH: Path = PROJECT_ROOT / "data"
    MODEL_PATH: Path = DATA_PATH / "models" / "best_model.pkl"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "Movie Recommendation API"
    API_VERSION: str = "1.0.0"
    
    # Model
    DEFAULT_N_RECOMMENDATIONS: int = 10
    MAX_N_RECOMMENDATIONS: int = 100
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "movie-recommendation"
    
    class Config:
        env_file = ".env"


settings = Settings()
