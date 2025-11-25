"""
Centralized logging configuration for the Air Quality Prediction backend.
Supports multi-level logging to both file and console with rotation.
"""

import logging
import logging.handlers
from pathlib import Path

# Safe log directory (relative to the working dir, always present in container/app)
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Centralized logs directory (also inside the app, just a subfolder)
CENTRALIZED_LOGS_DIR = LOGS_DIR / "backend"
CENTRALIZED_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Log file paths (local)
MAIN_LOG_FILE = LOGS_DIR / "app.log"
ERROR_LOG_FILE = LOGS_DIR / "error.log"
DEBUG_LOG_FILE = LOGS_DIR / "debug.log"
API_LOG_FILE = LOGS_DIR / "api.log"

# Log file paths (centralized)
MAIN_LOG_FILE_CENTRAL = CENTRALIZED_LOGS_DIR / "app.log"
ERROR_LOG_FILE_CENTRAL = CENTRALIZED_LOGS_DIR / "error.log"
DEBUG_LOG_FILE_CENTRAL = CENTRALIZED_LOGS_DIR / "debug.log"
API_LOG_FILE_CENTRAL = CENTRALIZED_LOGS_DIR / "api.log"

# Log format
DETAILED_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
SIMPLE_FORMAT = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Ensure log files exist
for log_file in [
    MAIN_LOG_FILE, ERROR_LOG_FILE, DEBUG_LOG_FILE, API_LOG_FILE,
    MAIN_LOG_FILE_CENTRAL, ERROR_LOG_FILE_CENTRAL, DEBUG_LOG_FILE_CENTRAL, API_LOG_FILE_CENTRAL
]:
    log_file.touch(exist_ok=True)

def setup_logger(name: str, log_file: Path = None, log_file_central: Path = None, level: int = logging.INFO,
                 console_output: bool = True, file_rotation: bool = True) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    Writes to both local and centralized log directories.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True  # Ensure propagation to root logger

    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()

    log_path = log_file or MAIN_LOG_FILE
    log_path_central = log_file_central or MAIN_LOG_FILE_CENTRAL

    # LOCAL FILE HANDLER
    try:
        local_handler = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        local_handler.setLevel(level)
        local_handler.setFormatter(DETAILED_FORMAT)
        logger.addHandler(local_handler)
    except Exception as e:
        print(f"❌ Error creating local file handler: {e}")

    # CENTRALIZED FILE HANDLER
    try:
        central_handler = logging.handlers.RotatingFileHandler(
            str(log_path_central),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        central_handler.setLevel(level)
        central_handler.setFormatter(DETAILED_FORMAT)
        logger.addHandler(central_handler)
    except Exception as e:
        print(f"❌ Error creating centralized file handler: {e}")

    # CONSOLE HANDLER
    if console_output:
        try:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(SIMPLE_FORMAT)
            logger.addHandler(console_handler)
        except Exception as e:
            print(f"❌ Error creating console handler: {e}")

    return logger

# Initialize main logger for the app
app_logger = setup_logger(__name__, log_file=MAIN_LOG_FILE, log_file_central=MAIN_LOG_FILE_CENTRAL, level=logging.INFO)

# Initialize error-specific logger
error_logger = setup_logger("error_logger", log_file=ERROR_LOG_FILE, log_file_central=ERROR_LOG_FILE_CENTRAL, level=logging.ERROR, console_output=False)

# Initialize debug logger (for development)
debug_logger = setup_logger("debug_logger", log_file=DEBUG_LOG_FILE, log_file_central=DEBUG_LOG_FILE_CENTRAL, level=logging.DEBUG, console_output=False)

# Initialize API request logger
api_logger = setup_logger("api_logger", log_file=API_LOG_FILE, log_file_central=API_LOG_FILE_CENTRAL, level=logging.INFO, console_output=False)

def get_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a logger with a specific name.
    """
    if name is None:
        return app_logger
    return setup_logger(name, log_file=MAIN_LOG_FILE, log_file_central=MAIN_LOG_FILE_CENTRAL, level=level)

if __name__ == "__main__":
    # Test logging
    app_logger.info("Testing app logger")
    error_logger.error("Testing error logger")
    debug_logger.debug("Testing debug logger")
    api_logger.info("Testing API logger")
    print(f"✓ Logs directory: {LOGS_DIR}")
    print(f"✓ Main log: {MAIN_LOG_FILE}")
    print(f"✓ Error log: {ERROR_LOG_FILE}")
    print(f"✓ Debug log: {DEBUG_LOG_FILE}")
    print(f"✓ API log: {API_LOG_FILE}")
