import logging
from pathlib import Path

import cv2
import numpy as np

from utils.logger import create_logger

logger = create_logger(name=__name__, level=logging.DEBUG)


class FakePicamera2:
    def __init__(self, path: Path):
        logger.debug(f"Using directory of images at {path}")
        self._image_index = 0
        self._max_image_index = 0
        self._path = path
        self._image_paths = []

    def start(self):
        logger.debug("Start FakePicamera2")
        logger.debug(f"Reset image index in directory to 0")
        self._image_index = 0
        logger.debug(f"Looking in {self._path} for .jpg")
        self._image_paths = list(self._path.glob("*.jpg"))
        self._max_image_index = len(self._image_paths) - 1
        logger.debug(f"Found {len(self._image_paths)} images")
        self._image_paths.sort()

    @property
    def image_index(self) -> int:
        """
        Get the current image index in the directory.

        :return: The current image index.
        """
        return self._image_index

    @image_index.setter
    def image_index(self, value: int):
        """
        Set the current image index in the directory.

        :param value: The new image index.
        """
        if value < 0:
            logger.warning(f"Requested image index {value} is less than 0, setting to "
                           f"0")
            self._image_index = 0
        elif value > self._max_image_index:
            logger.warning(f"Requested image index {value} is greater than the maximum "
                           f"index {self._max_image_index}, setting to maximum")
            self._image_index = self._max_image_index
        else:
            logger.debug(f"Setting image index to {value}")
            self._image_index = value

    def capture_array(self) -> np.ndarray:
        path = str(self._image_paths[self._image_index].expanduser().resolve())
        logger.debug(f"Reading {path} (index {self._image_index}) into camera")
        frame = cv2.imread(path)
        return frame

    def stop(self):
        logger.debug("Stop FakePicamera2")
