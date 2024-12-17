import logging
from dataclasses import dataclass
from itertools import product
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
                     top_n_confident: int = 5,
                     return_annotations: bool = False) -> list[GetPieceMatrixResult]:
    """
    Get the piece matrix from the chessboard only image.

    :param cb_only: Chessboard only image.
    :param top_n_confident: Number of top most confident chessboard arrangements.
    :param return_annotations: Whether to return an annotated image.
    :return: A GetPieceMatrixResult dataclass. pieces will be a 2D numpy array,
     first dimension is the row, second dimension is the column.
    """
    global colors
    chessboard_size = cb_only.shape[0]
    possible_piece_arrangements = []
    for y in range(8):
        for x in range(8):
            x0 = x * chessboard_size // 8
            y0 = y * chessboard_size // 8
            x1 = (x + 1) * chessboard_size // 8
            y1 = (y + 1) * chessboard_size // 8
            square = cb_only[y0:y1, x0:x1]
            classify_results = piece_model(square, imgsz=64, verbose=False)
            probs = classify_results[0].probs
            # class_name = piece_model.names[probs.top1]
            # pieces[y, x] = class_name

            possible_pieces = list(zip([piece_model.names[i] for i in probs.top5],
                                       np.array(probs.top5conf)))
            while sum(c for _, c in possible_pieces) > 0.9999 and len(
                    possible_pieces) > 1:
                possible_pieces.pop()
            # print(
            #     f"({x}, {y}) = {[f"{p} {c * 100:.2f}" for p, c in possible_pieces]}")
            possible_piece_arrangements.append(possible_pieces)

            # if return_annotations:
            #     cv2.rectangle(annotation, (x0 + 4, y0 + 4), (x1 - 4, y1 - 4),
            #                   (colors[class_name][2], colors[class_name][1],
            #                    colors[class_name][0]), 4)

    top_n = []
    for i, combination in enumerate(product(*possible_piece_arrangements)):
        if i == top_n_confident:
            break
        # print(f"Arrangement {i} {sum(c for _, c in combination) / 64}")
        top_n.append(combination)

    result = []
    for combo in top_n:
        pieces = np.zeros((8, 8), dtype=str)
        for y in range(8):
            for x in range(8):
                pieces[y, x] = combo[y * 8 + x][0]
        confidence = sum(c for _, c in combo) / 64
        annotation = None
        if return_annotations:
            annotation = cb_only.copy()
            for y in range(8):
                for x in range(8):
                    x0 = x * chessboard_size // 8
                    y0 = y * chessboard_size // 8
                    x1 = (x + 1) * chessboard_size // 8
                    y1 = (y + 1) * chessboard_size // 8
                    piece = combo[y * 8 + x][0]
                    cv2.rectangle(annotation, (x0 + 4, y0 + 4), (x1 - 4, y1 - 4),
                                  (colors[piece][2], colors[piece][1],
                                   colors[piece][0]), 4)

        result.append(
            GetPieceMatrixResult(pieces=pieces, confidence=confidence,
                                 annotation=annotation)
        )

    return result
