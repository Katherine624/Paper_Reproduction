#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends libosmesa6-dev libgl1-mesa-glx libegl1 libglew-dev libglfw3-dev libgles2-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6
