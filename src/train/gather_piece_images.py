import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Union

import cv2
import numpy as np
from dotenv import load_dotenv
from picamera2 import Picamera2
from roboflow import Roboflow
from shapely import Polygon
from ultralytics import YOLO

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
workspaceId = os.getenv("ROBOFLOW_WORKSPACE_ID")
projectId = os.getenv("ROBOFLOW_CHESSBOT_PIECES_PROJECT_ID")
print(f"Roboflow workspace ID: {workspaceId}")
print(f"Roboflow Chessbot Pieces project ID: {projectId}")
project = rf.workspace(workspaceId).project(projectId)

ncnn_path = Path.cwd() / "src" / "models" / "board_segmentation_best_ncnn_model"

model = YOLO(ncnn_path, task="segment")

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


def write_text_tl(image: np.ndarray, text: str):
    cv2.putText(image,
                text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)


def draw_polygon(pg: Union[Polygon, np.ndarray], image: np.ndarray, pt_size: int = 5,
                 pt_color: tuple = (0, 255, 0), line_color: tuple = (0, 255, 0),
                 line_width: int = 2) -> np.ndarray:
    if isinstance(pg, Polygon):
        pg = pg.exterior.coords
    for i, pt in enumerate(pg):
        x, y = pt
        cv2.circle(image, (int(x), int(y)), pt_size, pt_color)
        if i > 0:
            prev_x, prev_y = pg[i - 1]
            cv2.line(image,
                     (int(prev_x), int(prev_y)), (int(x), int(y)), line_color,
                     line_width)
        else:
            last_x, last_y = pg[-1]
            cv2.line(image,
                     (int(last_x), int(last_y)), (int(x), int(y)), line_color,
                     line_width)
    return image


def crop_and_reshape_to_square(image: np.ndarray, points: np.ndarray,
                               size: int = 500) -> np.ndarray:
    # Define the destination points for the square image
    dst_points = np.array([
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1],
        [0, 0]
    ], dtype="float32")

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(points, dst_points)

    # Apply the perspective transformation
    square_image = cv2.warpPerspective(image, M, (size, size))

    return square_image


def upload_np_image(image: np.ndarray):
    with NamedTemporaryFile(suffix=".jpg") as tmp:
        # print(f"Saving to {tmp.name}")
        cv2.imwrite(tmp.name, image)
        # print(f"Uploading to Roboflow")
        project.upload_image(tmp.name)


while True:
    frame = cam.capture_array()

    preview = frame.copy()
    chessboard_only = None
    chessboard_annotations = np.zeros((500, 500, 3), dtype="uint8")
    write_text_tl(chessboard_annotations, "No square")

    results = model(frame, verbose=False)
    if results[0].masks is not None:
        mask = results[0].masks.xy[0]

        pg = Polygon(mask).simplify(tolerance=30)
        preview = draw_polygon(pg, preview)

        rectangularity = pg.area / pg.minimum_rotated_rectangle.area
        has_four_pts = len(pg.exterior.coords) == 5
        write_text_tl(preview, f"{rectangularity:.2f}, {has_four_pts=}")

        if has_four_pts and rectangularity > 0.9:
            chessboard_only = crop_and_reshape_to_square(frame, np.array(
                [(pt[0], pt[1]) for pt in pg.exterior.coords][:4], dtype="float32"))

            chessboard_annotations = chessboard_only.copy()

            for i in range(1, 8):
                cv2.line(chessboard_annotations, (0, i * 500 // 8), (500, i * 500 // 8),
                         (0, 255, 0), 2)
                cv2.line(chessboard_annotations, (i * 500 // 8, 0), (i * 500 // 8, 500),
                         (0, 255, 0), 2)
    else:
        write_text_tl(preview, "No chessboard detected")

    cv2.imshow("Preview", preview)
    cv2.imshow("Chessboard only - C to capture", chessboard_annotations)

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "c":
        if chessboard_only is not None:
            captured_frame = chessboard_only.copy()
            cv2.imshow("Captured chessboard only - Y to save, N to discard",
                       captured_frame)
            print("Captured")
            print("Press Y to save, N to discard")
            while True:
                key = chr(cv2.waitKey(1) & 0xFF)
                if key == "y":
                    # Split captured_frame into 64 smaller images to upload
                    count = 1
                    for y in range(8):
                        for x in range(8):
                            x0 = x * 500 // 8
                            y0 = y * 500 // 8
                            x1 = (x + 1) * 500 // 8
                            y1 = (y + 1) * 500 // 8
                            square = captured_frame[y0:y1, x0:x1]
                            print(f"Uploading square {count}/64")
                            upload_np_image(square)
                            count += 1
                    print("Uploaded")
                elif key in ("n", "q"):
                    print("Discarded")
                else:
                    continue
                break
            cv2.destroyWindow("Captured chessboard only - Y to save, N to discard")
        else:
            print("No chessboard detected")
    if key == "q":
        print("Exiting")
        break

cv2.destroyAllWindows()
