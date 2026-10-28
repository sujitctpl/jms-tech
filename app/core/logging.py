import logging
import re
import sys
from app.core.config import settings

# Regex to sanitize credentials from URLs like rtsp://user:pass@host or http://user:pass@host
URL_CREDENTIALS_REGEX = re.compile(r"://([^:\s/]+):([^@\s/]+)@")


def sanitize_url(url: str | None) -> str:
    """Masks username and password in RTSP/HTTP/DB URLs."""
    if not url:
        return ""
    return URL_CREDENTIALS_REGEX.sub(r"://***:***@", str(url))


class SensitiveFilter(logging.Filter):
    """Logging filter to sanitize passwords/credentials from all log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = sanitize_url(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: sanitize_url(str(v)) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(sanitize_url(str(arg)) if isinstance(arg, str) else arg for arg in record.args)
        return True


def setup_logging() -> None:
    """Configures application-wide logging with sensitive data sanitization."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if already configured
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(SensitiveFilter())
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.addFilter(SensitiveFilter())


logger = logging.getLogger("showroom_vision")
