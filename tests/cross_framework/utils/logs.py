import os
import logging
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: str, name: str = "illia_tests") -> logging.Logger:
    """
    Set up logging with a rotating file handler.

    Args:
        log_dir: Directory where the log file will be stored.
        name: The name used for naming the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")

    # Set up rotating file handler (max 1MB per file, keep 5 backups)
    handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger
