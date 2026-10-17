#!/usr/bin/env bash
# Run the author's full LIBERO checkpoint; no LoRA is loaded.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$PROJECT_ROOT/upstream/cosmos-policy"
SUITE="${1:-libero_spatial}"
TRIALS="${2:-10}"
TASK_ID="${3:--1}"
RUN_NAME="${4:-official_${SUITE}_seed195_$(date +%Y%m%d_%H%M%S)}"
OUT="$PROJECT_ROOT/eval-results/$RUN_NAME"
if [[ -e "$OUT/started_at.txt" || -e "$OUT/console.log" ]]; then
    printf 'Run directory already contains results; choose a new RUN_NAME: %s\n' "$OUT" >&2
    exit 2
fi
mkdir -p "$OUT/videos" "$OUT/logs"
source "$REPO/.venv/bin/activate"
export BASE_DATASETS_DIR="$PROJECT_ROOT/datasets"
export HF_HOME="$PROJECT_ROOT/env/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export COSMOS_POLICY_BASE_MODEL_PATH="$PROJECT_ROOT/checkpoints/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"
export COSMOS_POLICY_SKIP_UNUSED_CHECKPOINT_DOWNLOADS=1
export COSMOS_POLICY_LOW_MEMORY_INFERENCE=1
export COSMOS_POLICY_EVAL_FAIL_FAST=1
export COSMOS_POLICY_VIDEO_DIR="$OUT/videos"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_HOME="$REPO/.venv/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$CUDA_HOME/cuda_nvrtc/lib:$CUDA_HOME/cudnn/lib:$CUDA_HOME/cublas/lib:$CUDA_HOME/cuda_runtime/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$REPO"
CKPT="$PROJECT_ROOT/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B"
CMD=(python -u -m cosmos_policy.experiments.robot.libero.run_libero_eval
    --config cosmos_predict2_2b_480p_libero__inference_only
    --config_file cosmos_policy/config/config.py
    --ckpt_path "$CKPT/Cosmos-Policy-LIBERO-Predict2-2B.pt"
    --tokenizer_path "$PROJECT_ROOT/checkpoints/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth"
    --dataset_stats_path "$CKPT/libero_dataset_statistics.json"
    --t5_text_embeddings_path "$CKPT/libero_t5_embeddings.pkl"
    --task_suite_name "$SUITE" --task_id "$TASK_ID" --num_trials_per_task "$TRIALS"
    --local_log_dir "$OUT/logs" --run_id_note "$RUN_NAME"
    --seed 195 --deterministic True --randomize_seed False
    --available_gpus 0 --use_parallel_inference False
    --use_wrist_image True --use_proprio True --normalize_proprio True
    --unnormalize_actions True --trained_with_image_aug True
    --chunk_size 16 --num_open_loop_steps 16 --data_collection False
    --use_variance_scale False --ar_future_prediction False --ar_value_prediction False
    --use_jpeg_compression True --flip_images True
    --num_denoising_steps_action 5 --num_denoising_steps_future_state 1 --num_denoising_steps_value 1)
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
date --iso-8601=seconds > "$OUT/started_at.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu.csv"
set +e
"${CMD[@]}" 2>&1 | tee "$OUT/console.log"
CODE=${PIPESTATUS[0]}
set -e
printf '%s\n' "$CODE" > "$OUT/exit_code.txt"
date --iso-8601=seconds > "$OUT/finished_at.txt"
exit "$CODE"
