# Chessbot

A Raspberry Pi that uses a camera to look at the board and tells you it's move
on a monitor.

## Installation

1. Have a Raspberry Pi (preferably 5) and a Raspberry Pi camera module and a
   monitor hooked up. Have the Raspberry Pi OS and desktop installed.
2. Update the Raspberry Pi. This step may be required if your camera feed is
   all messed up. (I wasted an entire day on this)
3. Use Python 3.11, at the time of writing some of the used packages are not
   compatible with Python 3.12.
4. Clone this repo.
5. Install [Stockfish](https://stockfishchess.org/) either with by downloading
   from the website or using `sudo apt install stockfish` on the Pi. (although
   it is several major versions behind it is better than compiling it)

There is an (untested) [`Makefile`](Makefile) available which performs the
rest of the installation steps. For the Pi, run `make install-for-pi` and for
Windows, run `make install-for-windows`. (If you don't have `make` installed,
you can run the commands manually)

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
images, which get uploaded to Roboflow automatically. Afterward, create a
dataset version
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

```commandline
python src/main.py --verbose
```

### Debugging

> Check the [`Makefile`](Makefile) for more `run-*` commands for development.

If you don't/can't use a camera, use the `--debug-use-image-dir` flag to use a
directory of static images, so you can run it on any computer. (The only
Raspberry Pi specific code is the camera access, which is replaced with image
reading code.)

```commandline
python src/main.py --debug-use-image-dir "test/scholars mate game white pov" --verbose 
```

The images should be 800x606, see
[`test/starting pos white pov`](test/starting%20pos%20white%20pov) for an
example. The images will be ordered by name, so best to order them with
numbers. (e.g. `01.png`, `02.png`, ...) Use <kbd>w</kbd> or <kbd>d</kbd> to
advance to the next image, and <kbd>s</kbd> or <kbd>a</kbd> to go to the
previous image. <kbd>q</kbd> to quit.

You can use the `--debug-play-image-dir` flag alongside the
`--debug-use-image-dir` flag to play the specified directory of images, like a
slideshow, in order to save some key pressing. Click any key in order to stop
the "slideshow".

You can capture your own image on the Raspberry Pi with a Picamera and
transfer it to your computer to be used with
[`test_camera.py`](src/train/test_camera.py):

```commandline
python src/train/test_camera.py  -d "/home/pi/Chessbot/test/new test"
```

Press <kbd>c</kbd> to save a numbered image to the directory specified with
`-d`. (It will start with `001.jpg`, then `002.jpg`, etc.)
