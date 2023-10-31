#!/bin/bash
set -x
source ~/mega_env/bin/activate
# export CUDA_HOME=/usr/local/cuda
cd ~/apex_good
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

# cd ~/TransformerEngine
# export NVIE_FRAMEWORK=pytorch
# pip install .