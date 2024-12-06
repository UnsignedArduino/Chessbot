import sys
from pathlib import Path

import cv2

from cv.board import GetChessboardOnlyResultType, get_chessboard_only
from utils.cv2_stuff import write_text_tl

sys.path.append(str(Path.cwd() / "src"))

from picamera2 import Picamera2

cam = Picamera2()
cam.configure(
    cam.create_video_configuration(main={"format": "RGB888", "size": (800, 606)}))
cam.start()

while True:
    frame = cam.capture_array()
    preview = frame.copy()

    result = get_chessboard_only(frame)
    if result.result_type == GetChessboardOnlyResultType.CHESSBOARD_FOUND:
        cb_only = result.chessboard
        preview = cb_only
    elif result.result_type == GetChessboardOnlyResultType.NO_CHESSBOARD_FOUND:
        write_text_tl(preview, "No chessboard found")
    elif result.result_type == GetChessboardOnlyResultType.NOT_QUADRILATERAL:
        write_text_tl(preview, "Chessboard not quadrilateral")
    elif result.result_type == GetChessboardOnlyResultType.NOT_RECTANGULAR_ENOUGH:
        write_text_tl(preview, "Chessboard not rectangular enough")

    cv2.imshow("Preview", preview)

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        print("Exiting")
        break

cv2.destroyAllWindows()
cam.stop()
