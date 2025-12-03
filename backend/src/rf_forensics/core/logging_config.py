"""
Centralized Logging Configuration

Unified logging setup for all RF Forensics components.
Supports structured logging, log rotation, and multiple output handlers.

Usage:
    from rf_forensics.core.logging_config import setup_logging, get_logger

    # Initialize logging once at application startup
    setup_logging(level="INFO", log_file="/var/log/rf_forensics.log")

    # Get logger in any module
    logger = get_logger(__name__)
    logger.info("Processing started", extra={"samples": 1000})
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# Custom Formatters
# =============================================================================


class StructuredFormatter(logging.Formatter):
    """
    JSON structured log formatter for production use.
    Includes timestamp, level, logger name, message, and any extra fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location info
        if record.pathname:
            log_entry["file"] = f"{record.pathname}:{record.lineno}"

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields passed via extra={}
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "message",
                "taskName",
            ):
                log_entry[key] = value

        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for development use.
    Uses ANSI color codes for better readability.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# =============================================================================
# Logger Setup Functions
# =============================================================================

_initialized = False


def setup_logging(
    level: str | int = "INFO",
    log_file: str | None = None,
    structured: bool = False,
    colored: bool = True,
    max_bytes: int = 10_000_000,  # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Initialize logging configuration for the entire application.

    Should be called once at application startup, typically in main().
    Subsequent calls will be ignored (idempotent).

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file for persistent logging
        structured: Use JSON structured format (good for log aggregation)
        colored: Use colored output for console (development)
        max_bytes: Max size of log file before rotation
        backup_count: Number of backup log files to keep
    """
    global _initialized
    if _initialized:
        return

    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers (prevents duplicate logs)
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if structured:
        console_handler.setFormatter(StructuredFormatter())
    elif colored and sys.stdout.isatty():
        fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        console_handler.setFormatter(ColoredFormatter(fmt, "%Y-%m-%d %H:%M:%S"))
    else:
        fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        console_handler.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))

    root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(level)

        # Always use structured format for file logs
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Set specific logger levels (reduce noise from noisy libraries)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("socketio").setLevel(logging.WARNING)
    logging.getLogger("engineio").setLevel(logging.WARNING)

    _initialized = True

    # Log initialization
    root_logger.info(
        "Logging initialized",
        extra={
            "level": logging.getLevelName(level),
            "log_file": log_file,
            "structured": structured,
        },
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing complete", extra={"count": 42})
    """
    return logging.getLogger(name)


def set_level(logger_name: str, level: str | int) -> None:
    """
    Set the logging level for a specific logger.

    Useful for adjusting verbosity at runtime.

    Args:
        logger_name: Name of the logger to configure
        level: New logging level
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger(logger_name).setLevel(level)


# =============================================================================
# Performance Logging
# =============================================================================


def log_performance(logger: logging.Logger, operation: str, duration_ms: float, **extra) -> None:
    """
    Log a performance metric.

    Args:
        logger: Logger instance
        operation: Name of the operation being measured
        duration_ms: Duration in milliseconds
        **extra: Additional context fields
    """
    logger.info(
        f"Performance: {operation}",
        extra={
            "operation": operation,
            "duration_ms": round(duration_ms, 2),
            "metric_type": "performance",
            **extra,
        },
    )


def log_throughput(
    logger: logging.Logger, metric_name: str, value: float, unit: str, **extra
) -> None:
    """
    Log a throughput metric.

    Args:
        logger: Logger instance
        metric_name: Name of the throughput metric
        value: Throughput value
        unit: Unit of measurement (e.g., "Msps", "MB/s")
        **extra: Additional context fields
    """
    logger.info(
        f"Throughput: {metric_name} = {value:.2f} {unit}",
        extra={
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "metric_type": "throughput",
            **extra,
        },
    )
