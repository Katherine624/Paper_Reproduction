source /root/gpufree-data/pi05-repro/env.sh
# The cloud image adds system cuDNN to LD_LIBRARY_PATH, mixing versions with uv.lock.
# Let JAX/PyTorch load the CUDA libraries installed in the project environment.
unset LD_LIBRARY_PATH
