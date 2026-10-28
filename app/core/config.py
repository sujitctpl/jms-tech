from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Showroom Visitor Intelligence - Phase 1"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Database Settings
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5432/jms_live"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_RECYCLE: int = 1800

    # Vision & Detection Settings
    YOLO_MODEL_PATH: str = "models/yolo11m.pt"
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.25
    PERSON_CLASS_ID: int = 0
    DEVICE: str = "cuda:0"
    TRACKER_TYPE: str = "bytetrack.yaml"

    # Worker Settings
    SAVE_DETECTION_EVENTS: bool = False
    FRAME_SKIP: int = 0
    RECONNECT_INTERVAL_SECONDS: int = 5
    MAX_RECONNECT_ATTEMPTS: int = 5

    # Logging
    LOG_LEVEL: str = "INFO"


settings = Settings()
