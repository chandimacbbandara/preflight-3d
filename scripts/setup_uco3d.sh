#!/bin/bash
set -e

# Activate your virtual environment first if using venv/conda
pip install --upgrade pip
pip install opencv-python-headless numpy pandas pyyaml tqdm trimesh

# Clone and install Meta's official uco3d loader & downloader outside data/
mkdir -p external
if [ ! -d "external/uco3d" ]; then
    git clone https://github.com/facebookresearch/uco3d.git external/uco3d
fi
pip install -e external/uco3d

echo "uCO3D toolkit installed successfully."