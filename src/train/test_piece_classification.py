from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2
from shapely.geometry.polygon import Polygon
from ultralytics import YOLO

board_segment_model_path = Path.cwd() / "src" / "models" / "board_segmentation_best.pt"
board_segment_ncnn_path = Path.cwd() / "src" / "models" / "board_segmentation_best_ncnn_model"

piece_classify_model_path = Path.cwd() / "src" / "models" / "piece_classification_best.pt"
piece_classify_ncnn_path = Path.cwd() / "src" / "models" / "piece_classification_best_ncnn_model"

if not board_segment_ncnn_path.exists():
    board_model = YOLO(board_segment_model_path)
    board_model.export(format="ncnn")

board_model = YOLO(board_segment_ncnn_path, task="segment")

if not piece_classify_ncnn_path.exists():
    piece_model = YOLO(piece_classify_model_path)
    piece_model.export(format="ncnn")

piece_model = YOLO(piece_classify_ncnn_path, task="classify")

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


def write_text_tl(image: np.ndarray, text: str):
    cv2.putText(image,
                text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)


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


chessboard_size = 512

while True:
    frame = cam.capture_array()
    segment_results = board_model(frame, verbose=False)
    if segment_results[0].masks is not None:
        mask = segment_results[0].masks.xy[0]

        pg = Polygon(mask).simplify(tolerance=50)

        rectangularity = pg.area / pg.minimum_rotated_rectangle.area
        has_four_pts = len(pg.exterior.coords) == 5

        if has_four_pts and rectangularity > 0.9:
            chessboard_only = crop_and_reshape_to_square(frame, np.array(
                [(pt[0], pt[1]) for pt in pg.exterior.coords][:4], dtype="float32"),
                                                         chessboard_size)

            for y in range(8):
                for x in range(8):
                    x0 = x * chessboard_size // 8
                    y0 = y * chessboard_size // 8
                    x1 = (x + 1) * chessboard_size // 8
                    y1 = (y + 1) * chessboard_size // 8
                    square = frame[y0:y1, x0:x1]
                    classify_results = piece_model(square, imgsz=64, verbose=False)
                    class_name = piece_model.names[classify_results[0].probs.top1]
                    colors = {
                        "bb": (255, 255, 0),  # yellow
                        "bk": (255, 0, 0),  # blue
                        "bn": (0, 255, 0),  # green
                        "bp": (127, 0, 255),  # purple
                        "bq": (255, 127, 0),  # orange
                        "br": (0, 192, 255),  # light blue
                        "empty": (0, 0, 0),  # black
                        "occluded": (127, 127, 127),  # gray
                        "wb": (255, 255, 96),  # light yellow
                        "wk": (255, 127, 127),  # light red
                        "wn": (127, 255, 127),  # light green
                        "wp": (192, 127, 255),  # light purple
                        "wq": (255, 192, 127),  # light orange
                        "wr": (127, 224, 255)  # light light blue
                    }
                    cv2.rectangle(chessboard_only, (x0, y0), (x1, y1),
                                  colors[class_name], 4)

            frame = chessboard_only
        else:
            write_text_tl(frame, "Not square")
    else:
        write_text_tl(frame, "No chessboard detected")

    cv2.imshow("Board segmentation and piece classification inference", frame)

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        print("Exiting")
        break

cv2.destroyAllWindows()
