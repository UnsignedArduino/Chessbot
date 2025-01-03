import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

import logging
from argparse import ArgumentParser

import cv2

from chessbot import Chessbot
from utils.cv2_stuff import write_text
from utils.logger import create_logger, set_all_stdout_logger_levels

logger = create_logger(name=__name__, level=logging.DEBUG)

parser = ArgumentParser(description="A Raspberry Pi that uses a camera to look at the "
                                    "board and tells you it's move on a monitor.")
parser.add_argument("--debug-use-image-dir", type=Path, default=None,
                    help="Use a directory of images instead of the camera as the "
                         "source.")
parser.add_argument("--debug-play-image-dir", action="store_true",
                    help="Play images in the specified directory instead of waiting "
                         "for key presses to cycle through. (like a slideshow)")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Enable verbose logging.")
args = parser.parse_args()
logger.debug(args)
if args.verbose:
    logger.debug("Enabling verbose logging")
    set_all_stdout_logger_levels(logging.DEBUG)

debug_play_image_dir = bool(args.debug_play_image_dir)

cam = None
debug_image_dir = Path(args.debug_use_image_dir) if args.debug_use_image_dir else None

if debug_play_image_dir and debug_image_dir is None:
    raise ValueError("You must specify a directory of images to play.")

if debug_image_dir is not None:
    logger.info(f"Using image directory {debug_image_dir} for debugging")

    from utils.fake_picamera2 import FakePicamera2

    cam = FakePicamera2(debug_image_dir)
    cam.start()
else:
    logger.debug("Using Picamera2")

    from picamera2 import Picamera2

    cam = Picamera2()
    cam.configure(
        cam.create_video_configuration(main={"format": "RGB888", "size": (800, 606)}))
    cam.start()

chessbot = Chessbot()

# For testing
# cam.image_index = 94  # start on white a couple before promotion
# frame = cam.capture_array()
# frame = cv2.flip(frame, 1)  # Flip horizontally
# chessbot.update(frame, force_board_sync=True)

while True:
    frame = cam.capture_array()
    frame = cv2.flip(frame, 1)  # Flip horizontally

    result = chessbot.update(frame)
    # logger.debug(f"Chessbot frame update result: {result}")

    cam_preview = chessbot.camera_preview
    if debug_image_dir is not None:
        write_text(cam_preview,
                   f"Image {cam.image_index + 1}/{cam.max_image_index + 1}", 10, 40)
    cv2.imshow("Camera preview", cam_preview)
    cv2.imshow("Chessboard preview", chessbot.chessboard_preview)

    key = ""
    if debug_image_dir is not None:
        logger.info("Use 'a' or 's' to go back, 'd' or 'w' to go forward, 'q' to quit.")
        if debug_play_image_dir:
            logger.info("Playing images in directory, automatically moving to the "
                        "next one. If a key specified above is pressed, automatic "
                        "playback will stop.")
            key = chr(cv2.waitKey(1) & 0xFF)
            pressed_key = False
            if key in ("d", "w"):
                cam.image_index += 1
                pressed_key = True
            elif key in ("a", "s"):
                cam.image_index -= 1
                pressed_key = True
            elif key == "q":
                pressed_key = True
                pass
            else:
                if cam.image_index < cam.max_image_index:
                    logger.debug("Automatically moving to next image now")
                    cam.image_index += 1
                else:
                    logger.debug("Reached end, stopping")
                    debug_play_image_dir = False
            if pressed_key:
                logger.info("Key pressed, stopping automatic playback")
                debug_play_image_dir = False
        else:
            while True:
                key = chr(cv2.waitKey(0) & 0xFF)
                if key in ("d", "w"):
                    cam.image_index += 1
                elif key in ("a", "s"):
                    cam.image_index -= 1
                elif key == "q":
                    pass
                else:
                    print("Unknown key")
                    continue
                break
    else:
        key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        logger.debug("Exiting")
        break

cv2.destroyAllWindows()
cam.stop()
chessbot.quit()
