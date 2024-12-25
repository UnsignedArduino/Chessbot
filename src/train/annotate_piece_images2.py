import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

import cv2

from tqdm import tqdm
from ultralytics import YOLO

from argparse import ArgumentParser

parser = ArgumentParser(description="Helps annotate piece images")
parser.add_argument("-i", "--input-image-dir", type=Path, default=None,
                    help="Directory of images to annotate.")
args = parser.parse_args()
print(args)

piece_classify_model_path = Path.cwd() / "src" / "models" / "piece_classification_best.pt"
piece_classify_ncnn_path = Path.cwd() / "src" / "models" / "piece_classification_best_ncnn_model"

piece_model = YOLO(piece_classify_ncnn_path, task="classify")

classes = {
    "b": "black/b",
    "k": "black/k",
    "n": "black/n",
    "p": "black/p",
    "q": "black/q",
    "r": "black/r",
    "empty": "empty",
    "occluded": "occluded",
    "B": "white/B",
    "K": "white/K",
    "N": "white/N",
    "P": "white/P",
    "Q": "white/Q",
    "R": "white/R"
}

image_dir = Path(args.input_image_dir)

for image_class in classes.values():
    (image_dir / image_class).mkdir(parents=True, exist_ok=True)


def annotate_and_move(img_p: Path):
    # Read
    square = cv2.imread(str(img_p))
    # Classify
    classify_results = piece_model(square, imgsz=64, verbose=False)
    probs = classify_results[0].probs
    class_name = piece_model.names[probs.top1]
    # Move the image to the correct directory
    img_p.rename(image_dir / classes[class_name] / img_p.name)


for image in tqdm(image_dir.glob("*.jpg"), desc="Annotating images"):
    annotate_and_move(image)
