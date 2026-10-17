set -euo pipefail
cd /root/gpufree-data/pi05-repro
source env-libero.sh
envs/libero/bin/python scripts/libero-render-smoke.py
