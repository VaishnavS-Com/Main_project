"""
=============================================================================
FILE: src/utils/logger.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis
=============================================================================

PURPOSE:
    A reusable logging system for the entire project.
    
    WHY DO WE NEED LOGGING?
    - print() statements disappear when your program runs. 
    - Logging SAVES messages to a file with timestamps.
    - When something breaks at 3am during training, you can read the log
      file and find out EXACTLY what failed and when.
    - Professional software ALWAYS uses logging, not print().

    LOGGING LEVELS (from least to most serious):
    DEBUG   → Very detailed info (for developers debugging)
    INFO    → General progress messages ("Starting training...")
    WARNING → Something unexpected but not fatal ("Low memory!")
    ERROR   → Something broke ("File not found!")
    CRITICAL→ System-level failure ("Out of disk space!")

AUTHOR: [Your Name]
DATE: 2026
=============================================================================
"""

import logging          # Python's built-in logging module
import os               # For file path operations
import sys              # For outputting to console
from datetime import datetime  # For timestamped log file names

# Import our central configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import LOGS_DIR, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a configured logger object.
    
    WHAT IS A "LOGGER"?
    Think of it like a smart printer that:
    1. Adds a timestamp to every message
    2. Labels each message with its severity level
    3. Prints to your screen AND saves to a file simultaneously
    
    PARAMETERS:
    -----------
    name : str
        The name of the module requesting the logger.
        Convention: pass __name__ (the module's own name).
        This helps identify WHERE in the code a log message came from.
    
    RETURNS:
    --------
    logging.Logger : A configured logger object ready to use.
    
    USAGE EXAMPLE:
    --------------
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting ECG preprocessing...")
    logger.warning("Signal contains noise!")
    logger.error("File not found: patient_01.csv")
    """
    
    # Step 1: Create the log file name with today's date
    # datetime.now().strftime() formats the current date/time as a string
    # "%Y%m%d" → "20260717" (Year Month Day)
    today = datetime.now().strftime("%Y%m%d")
    log_filename = os.path.join(LOGS_DIR, f"afib_project_{today}.log")
    
    # Step 2: Make sure the logs directory exists
    # exist_ok=True means "don't crash if directory already exists"
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Step 3: Get (or create) a logger with this name
    # logging.getLogger() is smart - if you call it twice with the same name,
    # it returns the SAME logger (not a duplicate)
    logger = logging.getLogger(name)
    
    # Step 4: Only add handlers if logger doesn't have any yet
    # This prevents duplicate messages if get_logger() is called multiple times
    if not logger.handlers:
        
        # Set the minimum log level (from config)
        # getattr() converts the string "INFO" to the constant logging.INFO
        logger.setLevel(getattr(logging, LOG_LEVEL))
        
        # Create a FORMATTER - this defines the visual layout of log messages
        # Example output: "2026-07-17 10:30:45 | INFO | preprocessing | Starting filter..."
        formatter = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT
        )
        
        # HANDLER 1: Console Handler (prints to your terminal screen)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)       # Show everything on console
        console_handler.setFormatter(formatter)
        
        # HANDLER 2: File Handler (saves to log file)
        file_handler = logging.FileHandler(
            filename=log_filename,
            mode='a',       # 'a' = append mode (don't overwrite old logs)
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Attach both handlers to our logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# TEST: Run this file directly to test the logger
# =============================================================================
if __name__ == "__main__":
    # Get a logger for testing
    logger = get_logger(__name__)
    
    # Test all log levels
    logger.debug("DEBUG: This is very detailed debug information")
    logger.info("INFO: ECG project logger initialized successfully")
    logger.warning("WARNING: Low memory detected - proceed with caution")
    logger.error("ERROR: This would indicate something went wrong")
    logger.critical("CRITICAL: This would be a system failure message")
    
    print(f"\nLog file saved to: {LOGS_DIR}")
    print("Logger test complete! Check your logs/ folder for the log file.")
