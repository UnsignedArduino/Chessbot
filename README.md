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

[![Try Model](https://app.roboflow.com/images/try-model-badge.svg)](https://universe.roboflow.com/unsignedarduino-9db8i/chessbot-boards/model/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UnsignedArduino/Chessbot/blob/main/src/train/train_board_segmentation.ipynb)

After configuring your dataset on Roboflow, use
[`gather_board_images.py`](src/train/gather_board_images.py) to gather board
images, which get uploaded to Roboflow. Afterward, create a dataset version
and train with
[`train_board_segmentation.ipynb`](src/train/train_board_segmentation.ipynb).
Upload the `best.pt` from the runs directory to the [`src/models`](src/models)
directory and rename it to
[`board_segmentation_best.pt`](src/models/board_segmentation_best.pt).

Afterward, run
[`test_board_segmentation.py`](src/train/test_board_segmentation.py). This step
is necessary as it exports the model to NCNN format. It will also preview
inference results on the live camera feed, which can be used to ensure the
model is working correctly.

#### Training pieces classification

WIP

## Usage

WIP
