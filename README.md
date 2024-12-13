# Chessbot

A Raspberry Pi that uses a camera to look at the board and tells you it's move
on a monitor.

## Installation

1. Have a Raspberry Pi (preferably 5) and a Raspberry Pi camera module and a
   monitor hooked up. Have the Raspberry Pi OS and desktop installed.
2. Install the required packages.
   ```bash
   sudo apt install python3-picamera2
   ```
3. Create the virtual environment and inherit system packages.
   ```bash
   python -m venv .venv --system-site-packages
   ``` 
4. Install [`requirements.txt`](requirements.txt) in the virtual environment.
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
5. Uninstall `opencv-python-headless` if installed. (It breaks usage of
   functions such as `imshow`...)
   ```bash
   pip uninstall opencv-python-headless
   ``` 
6. Copy [`.env.sample`](.env.sample) to `.env` and fill in the values if you
   want to train it yourself. (You will need a Roboflow account)

### Model training

The bot requires two models, one for segmenting the board and another for
classifying the pieces found on squares. This requires two Roboflow datasets
and ML models to be trained.

The camera should be facing the board down from above, centered on the four
middle central squares. See the board segmentation dataset to see the setup I
used.

#### Training board segmentation

[![View dataset on Roboflow](https://img.shields.io/badge/-View_dataset_on_Roboflow-gray?logo=roboflow&logoColor=%236706CE&labelColor=white&color=%236706CE)](https://universe.roboflow.com/unsignedarduino-9db8i/chessbot-boards)
[![View dataset on Kaggle](https://img.shields.io/badge/-View_dataset_on_Kaggle-gray?logo=kaggle&logoColor=%2320BEFF&labelColor=gray&color=blue)](https://www.kaggle.com/datasets/unsignedarduino/chessbot-boards)
[![Train on Colab](https://img.shields.io/badge/-Train_on_Colab-gray?logo=googlecolab&logoColor=%23F9AB00&labelColor=gray&color=blue)](https://colab.research.google.com/github/UnsignedArduino/Chessbot/blob/main/src/train/train_board_segmentation.ipynb)

After configuring your dataset on Roboflow, use
[`gather_board_images.py`](src/train/gather_board_images.py) to gather board
images, which get uploaded to Roboflow automatically. Afterward, create a dataset version
and train with
[`train_board_segmentation.ipynb`](src/train/train_board_segmentation.ipynb).
Download `best.pt` to the [`src/models`](src/models)
directory and rename it to
[`board_segmentation_best.pt`](src/models/board_segmentation_best.pt).

Afterward, run
[`test_board_segmentation.py`](src/train/test_board_segmentation.py). This step
is necessary as it exports the model to NCNN format. It will also preview
inference results on the live camera feed, which can be used to ensure the
model is working correctly.

#### Training pieces classification

[![View dataset on Roboflow](https://img.shields.io/badge/-View_dataset_on_Roboflow-gray?logo=roboflow&logoColor=%236706CE&labelColor=white&color=%236706CE)](https://universe.roboflow.com/unsignedarduino-9db8i/chessbot-pieces-qxp5p)
[![View dataset on Kaggle](https://img.shields.io/badge/-View_dataset_on_Kaggle-gray?logo=kaggle&logoColor=%2320BEFF&labelColor=gray&color=blue)](https://www.kaggle.com/datasets/unsignedarduino/chessbot-pieces)
[![Train on Colab](https://img.shields.io/badge/-Train_on_Colab-gray?logo=googlecolab&logoColor=%23F9AB00&labelColor=gray&color=blue)](https://colab.research.google.com/github/UnsignedArduino/Chessbot/blob/main/src/train/train_piece_classification.ipynb)

After configuring your dataset on Roboflow, use
[`gather_piece_images.py`](src/train/gather_piece_images.py) to gather board
images. Upload the directory to Roboflow. Afterward, create a dataset version
and train with
[
`train_piece_classification.ipynb`](src/train/train_piece_classification.ipynb).
Download `best.pt` to the [`src/models`](src/models)
directory and rename it to
[`piece_classification_best.pt`](src/models/piece_classification_best.pt).

Afterward, run
[`test_piece_classification.py`](src/train/test_piece_classification.py). This
step
is necessary as it exports the model to NCNN format. It will also preview
inference results on the live camera feed, which can be used to ensure the
model is working correctly.

## Usage

WIP
