import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

import cv2
import numpy as np

from picamera2 import Picamera2
from shapely.geometry.polygon import Polygon
from ultralytics import YOLO

from utils.cv2_stuff import crop_and_reshape_to_square, write_text_tl
from utils.math_stuff import find_closest_to_right_angles

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

chessboard_size = 512

while True:
    frame = cam.capture_array()
    segment_results = board_model(frame, verbose=False)
    if segment_results[0].masks is not None:
        mask = segment_results[0].masks.xy[0]

        pg = Polygon(mask).simplify(tolerance=20)
        rectangularity = pg.area / pg.minimum_rotated_rectangle.area

        pg = find_closest_to_right_angles(pg)
        corners = [(int(x), int(y)) for x, y in pg.exterior.coords][:-1]

        if rectangularity > 0.90 and len(corners) == 4:
            chessboard_only = crop_and_reshape_to_square(frame, np.array(
                [(pt[0], pt[1]) for pt in pg.exterior.coords][:4], dtype="float32"),
                                                         chessboard_size)

            for y in range(8):
                for x in range(8):
                    x0 = x * chessboard_size // 8
                    y0 = y * chessboard_size // 8
                    x1 = (x + 1) * chessboard_size // 8
                    y1 = (y + 1) * chessboard_size // 8
                    square = chessboard_only[y0:y1, x0:x1]
                    classify_results = piece_model(square, imgsz=64, verbose=False)
                    probs = classify_results[0].probs
                    class_name = piece_model.names[probs.top1]
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
                    cv2.rectangle(chessboard_only, (x0 + 4, y0 + 4), (x1 - 4, y1 - 4),
                                  (colors[class_name][2], colors[class_name][1],
                                   colors[class_name][0]), 4)

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
