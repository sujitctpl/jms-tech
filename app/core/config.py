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
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.28
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
    FACE_SIMILARITY_THRESHOLD: float = 0.40  # Calibrated operating threshold for high precision
    FACE_MATCH_MARGIN: float = 0.05  # Minimum difference between Top-1 and Top-2 distinct candidate matches
    FACE_MIN_SIZE: int = 16  # Minimum face width/height in pixels (calibrated for seated/back-row staff)
    FACE_MIN_CONFIDENCE: float = 0.45  # Minimum face detector confidence to reject ears/hair/monitors
    FACE_BLUR_THRESHOLD: float = 5.0  # Minimum Laplacian variance for blur filtering
    FACE_MAX_YAW: float = 65.0  # Maximum head yaw in degrees (calibrated for angled seating / walking)
    FACE_MAX_PITCH: float = 50.0  # Maximum head pitch in degrees (accommodates 35°-45° overhead CCTV)
    FACE_MAX_ROLL: float = 40.0  # Maximum head roll in degrees
    FACE_MIN_EYE_DIST: float = 2.5  # Minimum inter-ocular distance in pixels for distant 720p streams
    FACE_MIN_ILLUMINATION: float = 18.0  # Minimum average grayscale brightness (reject silhouettes)
    FACE_MAX_ILLUMINATION: float = 245.0  # Maximum average grayscale brightness (reject blown-out glare)
    EMPLOYEE_MIN_CONFIRM_FRAMES: int = 3  # Minimum confirmed frames before locking identity
    EMPLOYEE_VOTE_WINDOW: int = 6  # Temporal sliding window for employee voting buffer
    FACE_RECOGNITION_COOLDOWN_FRAMES: int = 5  # Cooldown between recognition attempts on unresolved tracks
    FACE_MAX_RECOGNITION_ATTEMPTS: int = 5  # Maximum attempts per track to avoid redundant inference

    # 14-Day Rolling Biometric Retention Policy Settings
    FACE_RETENTION_DAYS: int = 14  # Expire embeddings not recognized for 14 days
    RETENTION_CLEANUP_INTERVAL_HOURS: int = 24  # Run scheduled cleanup at least once daily
    RETENTION_AUTO_CLEANUP_ENABLED: bool = True  # Enable background scheduled cleanup


settings = Settings()
