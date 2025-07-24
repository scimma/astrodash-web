import logging
import logging.config
from logging.handlers import RotatingFileHandler
import os
from config.settings import get_settings, Settings

def init_logging(config: Settings = None):
    config = config or get_settings()
    LOG_DIR = config.log_dir
    LOG_FILE = os.path.join(LOG_DIR, "app.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
            },
            "json": {
                "format": '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": %(message)s}'
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": config.log_level,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": LOG_FILE,
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
                "level": config.log_level,
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": config.log_level,
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)
