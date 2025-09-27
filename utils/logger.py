"""
Logging configuration for Arabic STT Platform
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_file: str = None, level: str = 'INFO') -> logging.Logger:
    """
    Set up logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler (10MB max, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

class RequestLogger:
    """Custom logger for HTTP requests."""
    
    def __init__(self):
        self.logger = setup_logger('request', level='INFO')
    
    def log_request(self, method: str, path: str, status_code: int, 
                   processing_time: float = None, job_id: str = None):
        """Log HTTP request details."""
        
        message_parts = [f"{method} {path} - {status_code}"]
        
        if processing_time is not None:
            message_parts.append(f"({processing_time:.3f}s)")
        
        if job_id:
            message_parts.append(f"[Job: {job_id}]")
        
        message = " ".join(message_parts)
        
        if status_code >= 500:
            self.logger.error(message)
        elif status_code >= 400:
            self.logger.warning(message)
        else:
            self.logger.info(message)

class ProcessingLogger:
    """Custom logger for processing operations."""
    
    def __init__(self):
        self.logger = setup_logger('processing', level='DEBUG')
    
    def log_processing_start(self, job_id: str, filename: str, options: dict):
        """Log processing start."""
        self.logger.info(
            f"Processing started - Job: {job_id}, File: {filename}, Options: {options}"
        )
    
    def log_processing_step(self, job_id: str, step: str, details: str = ""):
        """Log processing step."""
        message = f"Job {job_id} - {step}"
        if details:
            message += f": {details}"
        self.logger.info(message)
    
    def log_processing_complete(self, job_id: str, processing_time: float, 
                              output_formats: list):
        """Log processing completion."""
        self.logger.info(
            f"Processing completed - Job: {job_id}, Time: {processing_time:.2f}s, "
            f"Formats: {output_formats}"
        )
    
    def log_processing_error(self, job_id: str, error: str, step: str = None):
        """Log processing error."""
        message = f"Processing failed - Job: {job_id}"
        if step:
            message += f" at step '{step}'"
        message += f" - Error: {error}"
        self.logger.error(message)

def setup_application_logging(log_dir: str = "logs", level: str = 'INFO'):
    """
    Set up comprehensive logging for the entire application.
    
    Args:
        log_dir: Directory for log files
        level: Global log level
    """
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Application logger
    app_logger = setup_logger(
        'arabic_stt',
        log_file=log_path / 'application.log',
        level=level
    )
    
    # Request logger
    request_logger = setup_logger(
        'requests',
        log_file=log_path / 'requests.log',
        level='INFO'
    )
    
    # Processing logger
    processing_logger = setup_logger(
        'processing',
        log_file=log_path / 'processing.log',
        level='DEBUG'
    )
    
    # Error logger
    error_logger = setup_logger(
        'errors',
        log_file=log_path / 'errors.log',
        level='ERROR'
    )
    
    # System logger for performance metrics
    system_logger = setup_logger(
        'system',
        log_file=log_path / 'system.log',
        level='INFO'
    )
    
    app_logger.info("Application logging configured successfully")
    
    return {
        'app': app_logger,
        'requests': request_logger,
        'processing': processing_logger,
        'errors': error_logger,
        'system': system_logger
    }

class PerformanceLogger:
    """Logger for performance metrics and system monitoring."""
    
    def __init__(self):
        self.logger = setup_logger('performance')
    
    def log_memory_usage(self, process_name: str, memory_mb: float):
        """Log memory usage."""
        self.logger.info(f"Memory usage - {process_name}: {memory_mb:.1f}MB")
    
    def log_processing_speed(self, job_id: str, duration: float, 
                           audio_duration: float):
        """Log processing speed metrics."""
        speed_ratio = audio_duration / duration if duration > 0 else 0
        self.logger.info(
            f"Processing speed - Job: {job_id}, "
            f"Audio: {audio_duration:.1f}s, Processing: {duration:.1f}s, "
            f"Ratio: {speed_ratio:.2f}x"
        )
    
    def log_model_load_time(self, model_size: str, load_time: float):
        """Log model loading time."""
        self.logger.info(f"Model load time - {model_size}: {load_time:.2f}s")
    
    def log_disk_usage(self, directory: str, used_mb: float, available_mb: float):
        """Log disk usage."""
        self.logger.info(
            f"Disk usage - {directory}: {used_mb:.1f}MB used, "
            f"{available_mb:.1f}MB available"
        )

# Global loggers for easy access
loggers = {}

def get_logger(name: str) -> logging.Logger:
    """Get or create logger by name."""
    if name not in loggers:
        loggers[name] = setup_logger(name)
    return loggers[name]

def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise
    return wrapper

# Silence noisy third-party loggers
def silence_noisy_loggers():
    """Reduce verbosity of third-party libraries."""
    noisy_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3.connectionpool',
        'faster_whisper',
        'transformers.tokenization_utils',
        'transformers.configuration_utils',
        'transformers.modeling_utils'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
