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
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.10
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

    # Face Embedding & Identity Recovery Settings
    ARCFACE_MODEL_PATH: str = "models/w600k_r50.onnx"
    FACE_SIMILARITY_THRESHOLD: float = 0.52  # Calibrated threshold (rejects impostor/noise matches below 0.50)
    FACE_MIN_SIZE: int = 24  # Minimum face width/height in pixels
    FACE_MIN_CONFIDENCE: float = 0.35  # Minimum face detector confidence
    FACE_BLUR_THRESHOLD: float = 12.0  # Minimum Laplacian variance for blur filtering
    FACE_RECOGNITION_COOLDOWN_FRAMES: int = 5  # Cooldown between recognition attempts on unresolved tracks
    FACE_MAX_RECOGNITION_ATTEMPTS: int = 5  # Maximum attempts per track to avoid redundant inference

    # 14-Day Rolling Biometric Retention Policy Settings
    FACE_RETENTION_DAYS: int = 14  # Expire embeddings not recognized for 14 days
    RETENTION_CLEANUP_INTERVAL_HOURS: int = 24  # Run scheduled cleanup at least once daily
    RETENTION_AUTO_CLEANUP_ENABLED: bool = True  # Enable background scheduled cleanup


settings = Settings()
