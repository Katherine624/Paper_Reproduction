set -euo pipefail
ROOT=/root/gpufree-data/openvla-repro
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/results/pipeline_failure.txt"; exit "$status"' ERR
while ! test -f "$ROOT/results/install_complete.txt"; do sleep 5; done
source "$ROOT/env.sh"
cd "$ROOT/openvla"
printf 'preflight\n' > "$ROOT/results/pipeline_stage.txt"
python "$ROOT/scripts/preflight.py" > "$ROOT/logs/preflight.log" 2>&1
python "$ROOT/scripts/wait_model.py" spatial
printf 'smoke_evaluation\n' > "$ROOT/results/pipeline_stage.txt"
python -u experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint "$ROOT/models/spatial" --task_suite_name libero_spatial --task_ids 0 --num_trials_per_task 3 --center_crop True --seed 7 --local_log_dir "$ROOT/results/eval-smoke" --run_id_note smoke > "$ROOT/logs/eval-smoke.log" 2>&1
python -c 'import glob,json,os; paths=glob.glob(os.environ["OPENVLA_ROOT"]+"/results/eval-smoke/*.jsonl"); rows=[json.loads(x) for p in paths for x in open(p)]; assert len(rows)==3 and not any(x["error"] for x in rows), rows; assert any(x["success"] for x in rows), "No successes in smoke; inspect before full evaluation"'
printf 'formal_evaluation\n' > "$ROOT/results/pipeline_stage.txt"
python -u experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint "$ROOT/models/spatial" --task_suite_name libero_spatial --num_trials_per_task 10 --center_crop True --seed 7 --local_log_dir "$ROOT/results/eval-spatial-100" --run_id_note spatial100 > "$ROOT/logs/eval-spatial-100.log" 2>&1
python "$ROOT/scripts/wait_model.py" base
printf 'training_100_steps\n' > "$ROOT/results/pipeline_stage.txt"
python -u "$ROOT/scripts/train_smoke.py" --model "$ROOT/models/base" --data "$ROOT/data" --out "$ROOT/results/train-smoke" --steps 100 --batch-size 2 > "$ROOT/logs/train-smoke.log" 2>&1
printf 'training_resume\n' > "$ROOT/results/pipeline_stage.txt"
python -u "$ROOT/scripts/train_smoke.py" --model "$ROOT/models/base" --data "$ROOT/data" --out "$ROOT/results/train-smoke" --steps 105 --batch-size 2 --resume "$ROOT/results/train-smoke/checkpoint-100" > "$ROOT/logs/train-resume.log" 2>&1
printf 'complete\n' > "$ROOT/results/pipeline_stage.txt"
date -Is > "$ROOT/results/pipeline_complete.txt"
