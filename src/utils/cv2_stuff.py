from typing import Union

import cv2
import numpy as np
from shapely.geometry.polygon import Polygon


def draw_polygon(pg: Union[Polygon, np.ndarray], image: np.ndarray, pt_size: int = 5,
                 pt_color: tuple = (0, 255, 0), line_color: tuple = (0, 255, 0),
                 line_width: int = 2) -> np.ndarray:
    if isinstance(pg, Polygon):
        pg = pg.exterior.coords
    for i, pt in enumerate(pg):
        x, y = pt
        cv2.circle(image, (int(x), int(y)), pt_size, pt_color)
        if i > 0:
            prev_x, prev_y = pg[i - 1]
            cv2.line(image,
                     (int(prev_x), int(prev_y)), (int(x), int(y)), line_color,
                     line_width)
        else:
            last_x, last_y = pg[-1]
            cv2.line(image,
                     (int(last_x), int(last_y)), (int(x), int(y)), line_color,
                     line_width)
    return image


def crop_and_reshape_to_square(image: np.ndarray, points: np.ndarray,
                               size: int) -> np.ndarray:
    # Define the destination points for the square image
    dst_points = np.array([
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1],
        [0, 0]
    ], dtype="float32")

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(points, dst_points)

    # Apply the perspective transformation
    square_image = cv2.warpPerspective(image, M, (size, size))

    return square_image


def get_tile_in_image(image: np.ndarray, row: int, col: int, rows: int = 8,
                      cols: int = 8) -> np.ndarray:
    x0 = col * image.shape[1] // cols
    y0 = row * image.shape[0] // rows
    x1 = (col + 1) * image.shape[1] // cols
    y1 = (row + 1) * image.shape[0] // rows
    return image[y0:y1, x0:x1]


font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 40)
fontScale = 1
fontColor = (255, 255, 255)
thickness = 3
lineType = 2


def write_text_tl(image: np.ndarray, text: str):
    cv2.putText(image,
                text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
