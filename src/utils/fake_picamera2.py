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
        logger.debug("Start")
        self.frame = cv2.imread(str(self.image_path.expanduser().resolve()))

    def capture_array(self):
        return self.frame

    def stop(self):
        logger.debug("Stop")
        self.frame = None
