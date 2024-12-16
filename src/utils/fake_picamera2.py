import logging
from pathlib import Path

import cv2

from utils.logger import create_logger

logger = create_logger(name=__name__, level=logging.DEBUG)


class FakePicamera2:
    def __init__(self, image_path: Path):
        logger.debug(f"Using saved image at {image_path}")
        self.image_path = image_path
        self.frame = None

    def start(self):
        logger.debug("Start FakePicamera2")
        p = str(self.image_path.expanduser().resolve())
        logger.debug(f"Reading image from {p}")
        self.frame = cv2.imread(p)

    def capture_array(self):
        return self.frame

    def stop(self):
        logger.debug("Stop FakePicamera2")
        self.frame = None
