import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from utils.logger import create_logger

logger = create_logger(name=__name__, level=logging.DEBUG)

piece_classify_ncnn_path = Path.cwd() / "src" / "models" / "piece_classification_best_ncnn_model"
logger.info(f"Loading piece classification model from {piece_classify_ncnn_path}")
piece_model = YOLO(piece_classify_ncnn_path, task="classify")

colors = {
    "b": (255, 255, 0),  # yellow
    "k": (255, 0, 0),  # blue
    "n": (0, 255, 0),  # green
    "p": (127, 0, 255),  # purple
    "q": (255, 127, 0),  # orange
    "r": (0, 192, 255),  # light blue
    "empty": (0, 0, 0),  # black
    "occluded": (127, 127, 127),  # gray
    "B": (255, 255, 96),  # light yellow
    "K": (255, 127, 127),  # light red
    "N": (127, 255, 127),  # light green
    "P": (192, 127, 255),  # light purple
    "Q": (255, 192, 127),  # light orange
    "R": (127, 224, 255)  # extra light blue
}


@dataclass
class GetPieceMatrixResult:
    pieces: np.ndarray
    confidence: float
    annotation: Optional[np.ndarray] = None


def get_piece_matrix(cb_only: np.ndarray,
                     return_annotations: bool = False) -> GetPieceMatrixResult:
    """
    Get the piece matrix from the chessboard only image.

    :param cb_only: Chessboard only image.
    :return: A GetPieceMatrixResult dataclass. pieces will be a 2D numpy array,
     first dimension is the row, second dimension is the column.
    """
    global colors
    pieces = np.zeros((8, 8), dtype=str)
    confidence = 0
    annotation = cb_only.copy() if return_annotations else None
    chessboard_size = cb_only.shape[0]
    for y in range(8):
        for x in range(8):
            x0 = x * chessboard_size // 8
            y0 = y * chessboard_size // 8
            x1 = (x + 1) * chessboard_size // 8
            y1 = (y + 1) * chessboard_size // 8
            square = cb_only[y0:y1, x0:x1]
            classify_results = piece_model(square, imgsz=64, verbose=False)
            probs = classify_results[0].probs
            class_name = piece_model.names[probs.top1]
            pieces[y, x] = class_name
            confidence += probs.top1conf / 64
            if return_annotations:
                cv2.rectangle(annotation, (x0 + 4, y0 + 4), (x1 - 4, y1 - 4),
                              (colors[class_name][2], colors[class_name][1],
                               colors[class_name][0]), 4)
    return GetPieceMatrixResult(pieces=pieces, confidence=confidence,
                                annotation=annotation)
