# Chessbot

A Raspberry Pi that uses a camera to look at the board and tells you it's move
on a monitor.

## Installation

1. Have a Raspberry Pi (preferably 5) and Raspberry Pi camera module and a
   monitor hooked up. Install the Raspberry Pi OS and desktop.
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
5. Uninstall `opencv-python-headless` if installed.
   ```bash
   pip uninstall opencv-python-headless
   ``` 
6. Copy [`.env.sample`](.env.sample) to `.env` and fill in the values if you
   want to train it yourself. (You will need a Roboflow account)

### Training

The bot requires two models, one for segmenting the board and another for
classifying each of the 64 squares within the board. This requires two Roboflow
datasets and ML models to be trained.

The camera should be facing the board down from above, centered on the four
middle central squares if possible.

#### Training board segmentation

To create your own data set, run [
`/src/train/gather_board_images.py`](src/train/gather_board_images.py),
which uploads captured images to the configured "Chessbot Boards" dataset on
Roboflow. (should be configured for segmentation)

```bash
python src/train/gather_board_images.py
```

If you are running the command over SSH, prepend `DISPLAY=:0` to the command.

To gather images:

* Click <kbd>c</kbd> to capture an image.
    * To save, click <kbd>y</kbd>. This uploads it to the configured Roboflow
      dataset.
    * To discard, click <kbd>n</kbd>.
* Click <kbd>q</kbd> to quit.

To annotate, segment by the four corners of the chess board.

#### Training square classification

WIP
