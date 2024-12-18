import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

import logging
from argparse import ArgumentParser

import chess.svg
import cv2

from cv.board import GetChessboardOnlyResultType, get_chessboard_only
from cv.pieces import get_piece_matrix
from utils.chess_stuff import board_sync_from_chessboard_arrangement
from utils.cv2_stuff import svg_to_numpy, write_text_tl
from utils.logger import create_logger, set_all_stdout_logger_levels

logger = create_logger(name=__name__, level=logging.DEBUG)

parser = ArgumentParser(description="A Raspberry Pi that uses a camera to look at the "
                                    "board and tells you it's move on a monitor.")
parser.add_argument("--debug-use-image", type=Path, default=None,
                    help="Use an image instead of the camera as the source")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Enable verbose logging")
args = parser.parse_args()
logger.debug(args)
if args.verbose:
    logger.debug("Enabling verbose logging")
    set_all_stdout_logger_levels(logging.DEBUG)

cam = None
debug_image = Path(args.debug_use_image) if args.debug_use_image else None

if debug_image is not None:
    logger.info(f"Using saved image {debug_image} for debugging")

    from utils.fake_picamera2 import FakePicamera2

    cam = FakePicamera2(debug_image)
    cam.start()
else:
    logger.debug("Using Picamera2")

    from picamera2 import Picamera2

    cam = Picamera2()
    cam.configure(
        cam.create_video_configuration(main={"format": "RGB888", "size": (800, 606)}))
    cam.start()

board = chess.Board()
board.clear()

while True:
    frame = cam.capture_array()
    frame = cv2.flip(frame, 1)  # Flip horizontally
    preview = frame.copy()

    result = get_chessboard_only(frame)
    cb_only = None
    if result.result_type == GetChessboardOnlyResultType.CHESSBOARD_FOUND:
        cb_only = result.chessboard
        preview = cb_only
    elif result.result_type == GetChessboardOnlyResultType.NO_CHESSBOARD_FOUND:
        write_text_tl(preview, "No chessboard found")
    elif result.result_type == GetChessboardOnlyResultType.NOT_QUADRILATERAL:
        write_text_tl(preview, "Chessboard not quadrilateral")
    elif result.result_type == GetChessboardOnlyResultType.NOT_RECTANGULAR_ENOUGH:
        write_text_tl(preview, "Chessboard not rectangular enough")

    unknown_squares = None
    if cb_only is not None:
        result = get_piece_matrix(cb_only, return_annotations=True)[0]
        preview = result.annotation
        write_text_tl(preview, f"{result.confidence:.4f}")
        unknown_squares = board_sync_from_chessboard_arrangement(board, result.pieces)
        # print(result.pieces)
        # print(str(board) == str(result.pieces))

    cv2.imshow("Camera preview", preview)
    cv2.imshow("Chessboard preview",
               svg_to_numpy(chess.svg.board(board, squares=unknown_squares, size=512)))

    if debug_image is not None:
        logger.info("Press any key to exit")
        cv2.waitKey(0)
        break
    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        logger.debug("Exiting")
        break

cv2.destroyAllWindows()
cam.stop()
