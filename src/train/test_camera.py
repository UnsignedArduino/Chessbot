from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2

parser = ArgumentParser(description="Tests the camera and optionally captures images.")
parser.add_argument("-d", "--directory", type=Path, default=None,
                    help="The dataset to save the images to. For example, /home/pi")
args = parser.parse_args()
print(args)

target_directory = None
if args.directory is not None:
    target_directory = Path(args.directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    print(f"Capturing images to {target_directory}")

file_idx = 1


def save_np_image(image: np.ndarray):
    global file_idx
    new_path = str(
        (target_directory / f"{file_idx:03}.jpg").expanduser().resolve())
    print(f"Saving to {new_path}")
    cv2.imwrite(new_path, image)
    file_idx += 1


cam = Picamera2()
cam.configure(
    cam.create_video_configuration(main={"format": "RGB888", "size": (800, 606)}))
cam.start()

while True:
    frame = cam.capture_array()

    cv2.imshow("Preview", frame)

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "c":
        if target_directory is not None:
            save_np_image(frame)
            print("Frame captured")
        else:
            print("No directory specified, cannot captured")
    if key == "q":
        print("Exiting")
        break

cv2.destroyAllWindows()
cam.stop()
