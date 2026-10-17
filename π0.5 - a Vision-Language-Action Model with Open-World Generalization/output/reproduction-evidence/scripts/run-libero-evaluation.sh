set -euo pipefail
cd /root/gpufree-data/pi05-repro
source env-libero.sh
export PYTHONUNBUFFERED=1
test -f results/train-smoke/restore-verification.json
envs/libero/bin/python scripts/evaluate_libero.py
