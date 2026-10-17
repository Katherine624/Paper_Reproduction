set -euo pipefail
cd /root/gpufree-data/pi05-repro
source env-model.sh
openpi/.venv/bin/python scripts/check-model-env.py
