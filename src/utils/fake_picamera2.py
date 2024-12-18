import logging
from pathlib import Path

import cv2
import numpy as np

from utils.logger import create_logger

logger = create_logger(name=__name__, level=logging.DEBUG)


class FakePicamera2:
    def __init__(self, path: Path):
        logger.debug(f"Using directory of images at {path}")
        self.image_index = 0
        self.max_image_index = 0
        self.path = path
        self.image_paths = []

    def start(self):
        logger.debug("Start FakePicamera2")
        logger.debug(f"Reset image index in directory to 0")
        self.image_index = 0
        logger.debug(f"Looking in {self.path} for .jpg")
        self.image_paths = list(self.path.glob("*.jpg"))
        self.max_image_index = len(self.image_paths) - 1
        logger.debug(f"Found {len(self.image_paths)} images")
        self.image_paths.sort()

    def capture_array(self) -> np.ndarray:
        path = str(self.image_paths[self.image_index].expanduser().resolve())
        logger.debug(f"Reading {path} (index {self.image_index}) into camera")
        frame = cv2.imread(path)
        if self.image_index < self.max_image_index:
            logger.debug("Will move to next frame")
            self.image_index += 1
        else:
            logger.debug("Already at last frame, will repeat this frame on next capture_    array")
        return frame

    def stop(self):
        logger.debug("Stop FakePicamera2")
