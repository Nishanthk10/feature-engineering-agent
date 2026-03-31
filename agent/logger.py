import logging
import os
from pathlib import Path

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # File handler — writes to outputs/agent.log
        fh = logging.FileHandler(OUTPUTS_DIR / "agent.log")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Console handler — prints WARNING and above
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger
