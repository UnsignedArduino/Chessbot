from pathlib import Path
from typing import Union

import cv2
import numpy as np
from picamera2 import Picamera2
from shapely import Polygon
from ultralytics import YOLO

ncnn_path = Path.cwd() / "src" / "models" / "board_segmentation_best_ncnn_model"

model = YOLO(ncnn_path, task="segment")

cam = Picamera2()
cam.configure(
    cam.create_video_configuration(main={"format": "RGB888", "size": (800, 606)}))
cam.start()

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 40)
fontScale = 1
fontColor = (255, 255, 255)
thickness = 3
lineType = 2


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
                               size: int = 500) -> np.ndarray:
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


while True:
    frame = cam.capture_array()

    results = model(frame, verbose=False)
    if results[0].masks is not None:
        mask = results[0].masks.xy[0]

        pg = Polygon(mask).simplify(tolerance=30)
        preview = draw_polygon(pg, frame.copy())

        rectangularity = pg.area / pg.minimum_rotated_rectangle.area
        has_four_pts = len(pg.exterior.coords) == 5
        cv2.putText(preview,
                    f"{rectangularity:.3f}, {has_four_pts=}",
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

        chessboard_only = crop_and_reshape_to_square(frame, np.array(
            [(pt[0], pt[1]) for pt in pg.exterior.coords][:4], dtype="float32"))

        cv2.imshow("Preview", preview)
        cv2.imshow("Chessboard only", chessboard_only)

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        print("Exiting")
        break

cv2.destroyAllWindows()
