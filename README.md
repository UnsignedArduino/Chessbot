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

To create your own data set, run [`/src/train/gather.py`](src/train/gather.py),
which uploads captured images to a configured dataset on Roboflow.

```bash
python src/train/gather.py
```

If you are running the command over SSH, prepend `DISPLAY=:0` to the command.

To gather images:

* Click <kbd>c</kbd> to capture an image.
    * To save, click <kbd>y</kbd>. This uploads it to the configured Roboflow
      dataset.
    * To discard, click <kbd>n</kbd>.
* Click <kbd>q</kbd> to quit.
