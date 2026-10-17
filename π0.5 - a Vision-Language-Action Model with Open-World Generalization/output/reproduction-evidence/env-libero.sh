source /root/gpufree-data/pi05-repro/env.sh
unset LD_LIBRARY_PATH
export LIBERO_CONFIG_PATH="$PI05_ROOT/config/libero"
export PYTHONPATH="$PI05_ROOT/openpi:$PI05_ROOT/openpi/packages/openpi-client/src:$PI05_ROOT/openpi/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
