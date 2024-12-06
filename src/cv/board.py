import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

sys.path.append(str(Path.cwd() / "src"))

import numpy as np

from shapely.geometry.polygon import Polygon
from ultralytics import YOLO

from utils.cv2_stuff import crop_and_reshape_to_square
from utils.math_stuff import find_closest_to_right_angles

board_segment_ncnn_path = Path.cwd() / "src" / "models" / "board_segmentation_best_ncnn_model"
board_model = YOLO(board_segment_ncnn_path, task="segment")

poly_simp_tolerance = 20
min_rectangularity = 0.9


class GetChessboardOnlyResultType(Enum):
    CHESSBOARD_FOUND = "CHESSBOARD_FOUND"
    NO_CHESSBOARD_FOUND = "NO_CHESSBOARD_FOUND"
    NOT_QUADRILATERAL = "NOT_QUADRILATERAL"
    NOT_RECTANGULAR_ENOUGH = "NOT_RECTANGULAR_ENOUGH"


@dataclass
class GetChessboardOnlyResult:
    result_type: GetChessboardOnlyResultType
    chessboard: Optional[np.ndarray] = None
    rectangularity: Optional[float] = None
    polygon: Optional[Polygon] = None


def get_chessboard_only(frame: np.ndarray,
                        chessboard_size: int = 512) -> GetChessboardOnlyResult:
    """
    Using a YOLOv11 model, segment the chessboard from the frame and return the
    chessboard only.

    :param frame: Camera input.
    :param chessboard_size: Output chessboard size, if detected. Defaults to 512.
    :return: A GetChessboardOnlyResult dataclass.
    """
    segment_results = board_model(frame, verbose=False)
    if segment_results[0].masks is not None:
        mask = segment_results[0].masks.xy[0]

        pg = Polygon(mask).simplify(tolerance=poly_simp_tolerance)
        rectangularity = pg.area / pg.minimum_rotated_rectangle.area

        pg = find_closest_to_right_angles(pg)
        corners = [(int(x), int(y)) for x, y in pg.exterior.coords][:-1]

        if rectangularity > min_rectangularity and len(corners) == 4:
            cb_only = crop_and_reshape_to_square(frame, np.array(
                [(pt[0], pt[1]) for pt in pg.exterior.coords][:4], dtype="float32"),
                                                 chessboard_size)
            return GetChessboardOnlyResult(
                result_type=GetChessboardOnlyResultType.CHESSBOARD_FOUND,
                chessboard=cb_only,
                rectangularity=rectangularity,
                polygon=pg
            )
        elif rectangularity <= min_rectangularity:
            return GetChessboardOnlyResult(
                result_type=GetChessboardOnlyResultType.NOT_RECTANGULAR_ENOUGH,
                rectangularity=rectangularity, polygon=pg)
        elif len(corners) != 4:
            return GetChessboardOnlyResult(
                result_type=GetChessboardOnlyResultType.NOT_QUADRILATERAL,
                rectangularity=rectangularity, polygon=pg)
    return GetChessboardOnlyResult(
        result_type=GetChessboardOnlyResultType.NO_CHESSBOARD_FOUND)
