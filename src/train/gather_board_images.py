import os
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from dotenv import load_dotenv
from picamera2 import Picamera2
from roboflow import Roboflow

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
workspaceId = os.getenv("ROBOFLOW_WORKSPACE_ID")
projectId = os.getenv("ROBOFLOW_CHESSBOT_BOARDS_PROJECT_ID")
print(f"Roboflow workspace ID: {workspaceId}")
print(f"Roboflow Chessbot Boards project ID: {projectId}")
project = rf.workspace(workspaceId).project(projectId)

cam = Picamera2()
cam.configure(
    cam.create_video_configuration(main={"format": "RGB888", "size": (800, 606)}))
cam.start()


def upload_np_image(image: np.ndarray):
    with NamedTemporaryFile(suffix=".jpg") as tmp:
        print(f"Saving to {tmp.name}")
        cv2.imwrite(tmp.name, image)
        print(f"Uploading to Roboflow")
        project.upload_image(tmp.name)


while True:
    original = cam.capture_array()
    cv2.imshow("Preview - C to capture", original)

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "c":
        cv2.imshow("Capture - Y to save, N to discard", original)
        while True:
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == "y":
                upload_np_image(original)
                print("Uploaded")
            elif key in ("n", "q"):
                print("Discarded")
            else:
                continue
            break
        cv2.destroyWindow("Capture - Y to save, N to discard")
    if key == "q":
        print("Exiting")
        break

cv2.destroyAllWindows()
