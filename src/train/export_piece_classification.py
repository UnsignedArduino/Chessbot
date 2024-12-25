from pathlib import Path

from ultralytics import YOLO

piece_classify_model_path = Path.cwd() / "src" / "models" / "piece_classification_best.pt"
piece_classify_ncnn_path = Path.cwd() / "src" / "models" / "piece_classification_best_ncnn_model"

if not piece_classify_ncnn_path.exists():
    print("Exporting piece classification model to NCNN format")
    piece_model = YOLO(piece_classify_model_path)
    piece_model.export(format="ncnn")
else:
    print("NCNN piece classification model already exists, skipping export")
