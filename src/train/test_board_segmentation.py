from pathlib import Path

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

ncnn_path = Path.cwd() / "src" / "models" / "board_segmentation_best_ncnn_model"

model = YOLO(ncnn_path, task="segment")

cam = Picamera2()
cam.configure(
    cam.create_video_configuration(main={"format": "RGB888", "size": (800, 606)}))
cam.start()

while True:
    frame = cam.capture_array()
    results = model(frame, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("Board segmentation inference", annotated)

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        print("Exiting")
        cv2.destroyAllWindows()
        break
