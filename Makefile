install-for-pi:
	echo "Installing for Raspberry Pi"

	sudo apt update
	sudo apt install python3-picamera2

	# Inherit system site packages for picamera2
	python -m venv .venv --system-site-packages
	source .venv/bin/activate
	# This may take a long time
	pip install -r requirements.txt
	# Breaks usage of imshow, etc.
	pip uninstall opencv-python-headless -y

	# Check if .env file exists, if not, copy .env.sample to .env
	# Only necessary if you want to train custom models
	@if [ ! -f .env ]; then cp .env.sample .env; echo "Copied .env.sample to .env" ; fi echo ".env already exists"

install-for-windows:
	echo "Installing for development on Windows"

	python -m venv .venv
	.venv\Scripts\activate
	# This may take a long time
	pip install -r requirements.txt
	# Breaks usage of imshow, etc.
	pip uninstall opencv-python-headless -y

activate-venv-for-pi:
	source .venv/bin/activate

activate-venv-for-windows:
	.venv\Scripts\activate

run-main:
	python src/main.py --verbose

run-main-scholars-mate-white-pov:
	python src/main.py --debug-use-image-dir "test/scholars mate game white pov" --verbose
