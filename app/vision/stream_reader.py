import time
from typing import Optional, Tuple
import cv2
import numpy as np
from app.core.config import settings
from app.core.logging import logger, sanitize_url


class StreamReader:
    """
    Robust video stream reader supporting RTSP streams and video files with auto-reconnection.
    """

    def __init__(
        self,
        stream_url: str,
        reconnect_interval: int = settings.RECONNECT_INTERVAL_SECONDS,
        max_reconnect_attempts: int = settings.MAX_RECONNECT_ATTEMPTS,
    ):
        self.stream_url = stream_url
        self.masked_url = sanitize_url(stream_url)
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self._consecutive_failures = 0

    def connect(self) -> bool:
        """Attempts to open the video stream."""
        self.release()
        logger.info("Opening video stream: %s", self.masked_url)
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if self.cap.isOpened():
                # Minimize RTSP buffer lag
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.is_connected = True
                self._consecutive_failures = 0
                logger.info("Stream opened successfully: %s", self.masked_url)
                return True
            else:
                logger.warning("Failed to open stream: %s", self.masked_url)
                self.is_connected = False
                return False
        except Exception as e:
            logger.error("Error opening stream %s: %s", self.masked_url, str(e))
            self.is_connected = False
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads next frame. Automatically handles reconnections if disconnected.
        Returns (success: bool, frame: Optional[np.ndarray]).
        """
        if not self.is_connected or self.cap is None or not self.cap.isOpened():
            if not self.reconnect():
                return False, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3:
                    logger.warning(
                        "Multiple failed frame reads on stream %s. Attempting reconnection...",
                        self.masked_url,
                    )
                    self.is_connected = False
                    if not self.reconnect():
                        return False, None
                return False, None

            self._consecutive_failures = 0
            return True, frame
        except Exception as e:
            logger.error("Exception reading frame from stream %s: %s", self.masked_url, str(e))
            self.is_connected = False
            return False, None

    def reconnect(self) -> bool:
        """Attempts to reconnect to stream up to max_reconnect_attempts."""
        attempts = 0
        while attempts < self.max_reconnect_attempts:
            attempts += 1
            logger.info(
                "Reconnection attempt %d/%d for %s in %ds...",
                attempts,
                self.max_reconnect_attempts,
                self.masked_url,
                self.reconnect_interval,
            )
            time.sleep(self.reconnect_interval)
            if self.connect():
                return True
        logger.error("Exceeded maximum reconnection attempts for stream %s", self.masked_url)
        return False

    def release(self) -> None:
        """Closes video capture."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_connected = False
