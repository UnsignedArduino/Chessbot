import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

import tempfile

from argparse import ArgumentParser
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from picamera2 import Picamera2
from shapely import Polygon
from ultralytics import YOLO

from utils.cv2_stuff import crop_and_reshape_to_square, draw_polygon, \
    get_tile_in_image, write_text
from utils.math_stuff import find_closest_to_right_angles

parser = ArgumentParser(description="Helps gather piece images")
parser.add_argument("-p", "--piece", choices=[
    "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k", "empty", "occluded"
], required=True, help="The class to gather images for")
parser.add_argument("-d", "--directory", type=Path, required=True,
                    help="The dataset to save the images to. For example, "
                         "/home/pi/Chessbot-Pieces-Dataset with piece P would save "
                         "images to ~/Chessbot-Pieces-Dataset/P")
args = parser.parse_args()
print(args)

target_piece = args.piece
target_directory = Path(args.directory) / target_piece
target_directory.mkdir(parents=True, exist_ok=True)

print(f"Capturing images for {target_piece} to {target_directory}")

ncnn_path = Path.cwd() / "src" / "models" / "board_segmentation_best_ncnn_model"

model = YOLO(ncnn_path, task="segment")

cam = Picamera2()
cam.configure(
    cam.create_video_configuration(main={"format": "RGB888", "size": (800, 606)}))
cam.start()

tempfile.tempdir = target_directory


def save_np_image(image: np.ndarray):
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        print(f"Saving to {tmp.name}")
        cv2.imwrite(tmp.name, image)


chessboard_size = 512
current_square = 0
print(f"Place {target_piece} on {current_square % 8}, {current_square // 8}")

# idfk why this is needed
try:
    model(np.zeros((640, 640, 3), dtype=np.uint8))
except TypeError:
    print("Model loaded")
    pass
model(np.zeros((640, 640, 3), dtype=np.uint8))

while True:
    frame = cam.capture_array()

    preview = frame.copy()
    chessboard_only = None
    chessboard_annotations = None

    results = model(frame, verbose=False)
    if results[0].masks is not None:
        mask = results[0].masks.xy[0]

        pg = Polygon(mask).simplify(tolerance=20)
        rectangularity = pg.area / pg.minimum_rotated_rectangle.area
        preview = draw_polygon(pg, preview)

        pg = find_closest_to_right_angles(pg)
        draw_polygon(pg, preview, pt_color=(0, 0, 255), line_color=(0, 0, 255))
        corners = [(int(x), int(y)) for x, y in pg.exterior.coords][:-1]

        write_text(preview, f"{rectangularity:.3f}, {len(corners)}", 10, 10)

        if rectangularity > 0.90 and len(corners) == 4:
            chessboard_only = crop_and_reshape_to_square(frame, np.array(corners,
                                                                         dtype="float32"),
                                                         chessboard_size)

            chessboard_annotations = chessboard_only.copy()

            for i in range(1, 8):
                cv2.line(chessboard_annotations, (0, i * chessboard_size // 8),
                         (chessboard_size, i * chessboard_size // 8),
                         (0, 255, 0), 2)
                cv2.line(chessboard_annotations, (i * chessboard_size // 8, 0),
                         (i * chessboard_size // 8, chessboard_size),
                         (0, 255, 0), 2)

            cv2.rectangle(chessboard_annotations, (
                current_square % 8 * chessboard_size // 8,
                current_square // 8 * chessboard_size // 8),
                          ((current_square % 8 + 1) * chessboard_size // 8,
                           (current_square // 8 + 1) * chessboard_size // 8),
                          (0, 0, 255), 4)

    cv2.imshow("Preview - C to capture",
               chessboard_annotations if chessboard_annotations is not None else preview)

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "c":
        if chessboard_only is not None:
            square = get_tile_in_image(chessboard_only, current_square // 8,
                                       current_square % 8)
            cv2.imshow("Capture - Y to save, N to discard", square)
            print("Captured")
            print("Press Y to save, N to discard")
            while True:
                key = chr(cv2.waitKey(1) & 0xFF)
                if key == "y":
                    print(
                        f"Saving at square {current_square % 8}, {current_square // 8}")
                    save_np_image(square)
                    current_square += 3
                    print("Saved")
                elif key in ("n", "q"):
                    print("Discarded")
                else:
                    continue
                break
            print(
                f"Place {target_piece} on {current_square % 8}, {current_square // 8}")
            cv2.destroyWindow("Capture - Y to save, N to discard")
        else:
            print("No chessboard detected")
    if key == "q":
        print("Exiting")
        break

cv2.destroyAllWindows()
