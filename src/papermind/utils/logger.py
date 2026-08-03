"""Custom logging utility using Loguru."""

import sys
from loguru import logger

# Remove default logger handlers
logger.remove()

# Add standard console output handler with clean formatting
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

__all__ = ["logger"]
