from functools import lru_cache
from io import BytesIO, StringIO
from typing import Union

import cv2
import numpy as np
from reportlab.graphics import renderPM
from shapely.geometry.polygon import Polygon
from svglib.svglib import svg2rlg


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


def write_text(image: np.ndarray, text: str, x: int = 10, y: int = 10):
    """
    Write text on an image.

    :param image: Numpy image to write to.
    :param text: Text to write.
    :param x: Left coordinate of the text. (x) Defaults to 10.
    :param y: Top coordinate of the text. (y) Defaults to 10.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 3
    line_type = 2

    (width, height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    bottom_left_corner_of_text = (x, height + y)

    text_area = image[y:y + height, x:x + width]
    brightness = np.mean(text_area)

    font_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

    cv2.putText(image,
                text,
                bottom_left_corner_of_text,
                font,
                font_scale,
                font_color,
                thickness,
                line_type)


@lru_cache(maxsize=32)
def svg_to_numpy(svg: str) -> np.ndarray:
    drawing = svg2rlg(StringIO(svg))
    png_buf = BytesIO()
    renderPM.drawToFile(drawing, png_buf, fmt="PNG")
    png_buf.seek(0)
    return cv2.imdecode(np.frombuffer(png_buf.read(), np.uint8), cv2.IMREAD_COLOR)
