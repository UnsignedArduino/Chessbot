import os
from tempfile import NamedTemporaryFile
from time import time as unix

import cv2
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

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 40)
fontScale = 1
fontColor = (255, 255, 255)
thickness = 3
lineType = 2

captured_frame = None
message = ""
message_time = 0

while True:
    original = cam.capture_array()

    if captured_frame is not None:
        frame = captured_frame.copy()
        cv2.putText(frame, "Click Y to upload, N to discard",
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        cv2.imshow("Preview", frame)
    else:
        frame = original.copy()
        if message != "":
            if unix() - message_time > 2:
                message = ""
        cv2.putText(frame, "Click C to capture" if message == "" else message,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        cv2.imshow("Preview", frame)

    key = chr(cv2.waitKey(1) & 0xFF)

    if captured_frame is not None:
        if key == "y":
            with NamedTemporaryFile(suffix=".jpg") as tmp:
                print(f"Saving to {tmp.name}")
                cv2.imwrite(tmp.name, captured_frame)
                print(f"Uploading to Roboflow")
                project.upload_image(tmp.name)
            print("Uploaded")
            message = "Uploaded"
            message_time = unix()
            captured_frame = None
        elif key == "n":
            print("Discarded")
            message = "Discarded"
            message_time = unix()
            captured_frame = None
    else:
        if key == "c":
            print("Captured")
            print("Press Y to save, N to discard")
            captured_frame = original.copy()
    if key == "q":
        print("Exiting")
        cv2.destroyAllWindows()
        break
