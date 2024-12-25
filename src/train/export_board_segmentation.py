from pathlib import Path

from ultralytics import YOLO

model_path = Path.cwd() / "src" / "models" / "board_segmentation_best.pt"
ncnn_path = Path.cwd() / "src" / "models" / "board_segmentation_best_ncnn_model"

if not ncnn_path.exists():
    print("Exporting model to NCNN format")
    model = YOLO(model_path)
    model.export(format="ncnn")
else:
    print("NCNN model already exists, skipping export")
