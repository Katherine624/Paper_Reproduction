set -euo pipefail
cd /root/gpufree-data/pi05-repro
source env-model.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONUNBUFFERED=1
ulimit -c 0
test -f checkpoints/pi05_base/server-verified.txt
openpi/.venv/bin/python scripts/pi05_experiment.py train
openpi/.venv/bin/python scripts/pi05_experiment.py restore
