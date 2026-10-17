set -euo pipefail
cd /root/gpufree-data/pi05-repro
source env.sh
test -x envs/libero/bin/python || uv venv --python 3.8 envs/libero
uv pip install --python envs/libero/bin/python -r openpi/examples/libero/requirements.txt -r openpi/third_party/libero/requirements.txt -r openpi/packages/openpi-client/pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip check --python envs/libero/bin/python
