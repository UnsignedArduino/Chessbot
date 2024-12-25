import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

import tempfile
from argparse import ArgumentParser
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from shapely import Polygon
from tqdm import trange
from ultralytics import YOLO

from utils.cv2_stuff import crop_and_reshape_to_square, draw_polygon, \
    get_tile_in_image, write_text
from utils.math_stuff import find_closest_to_right_angles

parser = ArgumentParser(description="Helps gather piece images")
parser.add_argument("-i", "--input-image-dir", type=Path, default=None,
                    help="Use a directory of images instead of the camera as the "
                         "source.")
parser.add_argument("-o", "--output-image-dir", type=Path, required=True,
                    help="The dataset to save the images to. For example, "
                         "/home/pi/Chessbot-Pieces-Dataset/test")
parser.add_argument("-a", "--auto", action="store_true",
                    help="Automatically capture every image in the input directory.")
args = parser.parse_args()
print(args)

input_image_dir = Path(args.input_image_dir)
output_image_dir = Path(args.output_image_dir)
output_image_dir.mkdir(parents=True, exist_ok=True)

auto = bool(args.auto)

print(f"Capturing images to {output_image_dir}")

ncnn_path = Path.cwd() / "src" / "models" / "board_segmentation_best_ncnn_model"

model = YOLO(ncnn_path, task="segment")

from utils.fake_picamera2 import FakePicamera2

cam = FakePicamera2(input_image_dir)
cam.start()

tempfile.tempdir = output_image_dir


def save_np_image(image: np.ndarray):
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, image)


chessboard_size = 512

# idfk why this is needed
try:
    model(np.zeros((640, 640, 3), dtype=np.uint8))
except TypeError:
    print("Model loaded")
    pass
model(np.zeros((640, 640, 3), dtype=np.uint8))

while True:
    frame = cam.capture_array()
    frame = cv2.flip(frame, 1)  # Flip horizontally

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

    cv2.imshow("Preview - C to capture",
               chessboard_annotations if chessboard_annotations is not None else preview)

    key = ""
    while True:
        if auto:
            cv2.waitKey(1)
            key = "c"
        else:
            key = chr(cv2.waitKey(0) & 0xFF)
        if key in ("d", "w"):
            cam.image_index += 1
        elif key in ("a", "s"):
            cam.image_index -= 1
        elif key == "c":
            print("Saving squares")
            for i in trange(64):
                square = get_tile_in_image(chessboard_only, i // 8, i % 8)
                save_np_image(square)
            print("Done, automatically moving to next image")
            if cam.image_index < cam.max_image_index:
                cam.image_index += 1
            else:
                print("Reached end, stopping")
                key = "q"
                break
        elif key == "q":
            pass
        else:
            print("Unknown key")
            continue
        break
    if key == "q":
        print("Exiting")
        break

cv2.destroyAllWindows()
