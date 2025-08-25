# src/logger_config.py

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

# Import the root directory from our config file
from src.config import ROOT_DIR

# --- LOGGER CONFIGURATION ---

# Define the format for log messages
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True) # Create the logs directory if it doesn't exist
LOG_FILE = LOG_DIR / "app.log"

def setup_logger(logger_name: str = "ml_app") -> logging.Logger:
    """
    Sets up and returns a configured logger.

    Args:
        logger_name (str): The name for the logger.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Get the logger instance
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO) # Set the minimum level of messages to log

    # Prevent messages from being duplicated in parent loggers
    logger.propagate = False

    # If handlers are already present, do not add them again
    if logger.hasHandlers():
        return logger

    # --- Handlers ---
    # A handler for writing to the console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # A handler for writing to a log file, with daily rotation
    # This creates a new log file every day and keeps the last 7 as backup.
    file_handler = TimedRotatingFileHandler(
        LOG_FILE, when="midnight", interval=1, backupCount=7, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Create a default logger instance to be imported by other modules
logger = setup_logger()