import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_ROOT / "upstream" / "cosmos-policy"
DATASETS_DIR = PROJECT_ROOT / "datasets"
HF_CACHE_DIR = PROJECT_ROOT / "env" / "hf-cache"


def main():
    env = os.environ.copy()
    env["BASE_DATASETS_DIR"] = str(DATASETS_DIR)
    env["HF_HOME"] = str(HF_CACHE_DIR)
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=8",
        "--master_port=12341",
        "-m", "cosmos_policy.scripts.train",
        "--config=cosmos_policy/config/config.py",
        "--",
        "experiment=cosmos_predict2_2b_480p_libero",
        "trainer.grad_accum_iter=8",
    ]

    print("开始 LIBERO 训练，工作目录：", REPO_DIR)
    subprocess.run(command, cwd=REPO_DIR, env=env, check=True)


if __name__ == "__main__":
    main()