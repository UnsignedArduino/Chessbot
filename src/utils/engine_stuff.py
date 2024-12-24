import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from utils.logger import create_logger

logger = create_logger(name=__name__, level=logging.DEBUG)


def find_stockfish_binary(recursive: bool = False) -> Optional[Path]:
    """
    Find the stockfish binary on the system. Searches the current directory and PATH.

    :param recursive: Whether to search recursively in the PATH.
    :return: The path to the stockfish binary, or None if it was not found.
    """
    logger.debug("Searching for stockfish binary")

    logger.debug(f"Searching current working directory {Path.cwd()}")
    for path in Path.cwd().rglob("stockfish*") if recursive else Path.cwd().glob(
            "stockfish*"):
        if path.is_file():
            logger.debug(f"Found stockfish binary at {path}")
            return path

    if (path := shutil.which("stockfish")) is not None:
        logger.debug(f"Found stockfish binary at {path}")
        return Path(path)

    for dir in [Path(p) for p in os.environ["PATH"].split(os.pathsep)]:
        logger.debug(f"Searching {dir}")
        for path in dir.rglob("stockfish*") if recursive else dir.glob("stockfish*"):
            if path.is_file():
                logger.debug(f"Found stockfish binary at {path}")
                return path

    logger.warning("Could not find stockfish binary")
    return None
