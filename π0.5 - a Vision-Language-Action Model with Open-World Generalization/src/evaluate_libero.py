"""Run upstream LIBERO evaluation; add per-episode records and unique video names."""
import collections
import json
import logging
import os
from pathlib import Path
import time
from examples.libero import main as upstream

ROOT = Path(os.environ["PI05_ROOT"])
OUT = ROOT / "results/eval-libero-spatial-seed7-trials10"
OUT.mkdir(parents=True, exist_ok=False)
ARGS = upstream.Args(host="127.0.0.1", port=8765, task_suite_name="libero_spatial",
                     num_trials_per_task=10, seed=7, video_out_path=str(OUT / "videos"))
RECORDS = []
START = time.time()
ENV = None
ORIGINAL_VIDEO = upstream.imageio.mimwrite
ORIGINAL_ENV = upstream._get_libero_env


def save_summary(completed=False):
    tasks = collections.defaultdict(lambda: {"episodes": 0, "successes": 0})
    for row in RECORDS:
        tasks[row["task"]]["episodes"] += 1
        tasks[row["task"]]["successes"] += int(row["success"])
    total = len(RECORDS)
    successes = sum(int(row["success"]) for row in RECORDS)
    report = {"completed": completed, "suite": ARGS.task_suite_name, "seed": ARGS.seed,
              "trials_per_task": ARGS.num_trials_per_task,
              "checkpoint": "official pi05_libero (independent of short training)",
              "episodes": total, "successes": successes,
              "success_rate": successes / total if total else None,
              "elapsed_seconds": time.time() - START, "per_task": dict(tasks),
              "openpi_commit": (ROOT / "logs/openpi-commit.txt").read_text().strip(),
              "rendering": "OSMesa CPU offscreen", "replan_steps": ARGS.replan_steps}
    temp = OUT / "summary.json.tmp"
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(OUT / "summary.json")


def recorded_video(path, images, **kwargs):
    original = Path(path)
    episode = len(RECORDS)
    target = original.with_name("episode_%03d_%s" % (episode, original.name))
    ORIGINAL_VIDEO(target, images, **kwargs)
    success = original.stem.endswith("_success")
    task = original.stem[len("rollout_"):].rsplit("_", 1)[0].replace("_", " ")
    row = {"episode_index": episode, "task_index": episode // ARGS.num_trials_per_task,
           "trial_index": episode % ARGS.num_trials_per_task, "task": task,
           "seed": ARGS.seed, "success": success, "recorded_control_frames": len(images),
           "video": str(target.relative_to(OUT)), "elapsed_seconds": time.time() - START}
    RECORDS.append(row)
    with (OUT / "episodes.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    save_summary()
    print("EPISODE_RECORDED", json.dumps(row), flush=True)


def managed_env(*args, **kwargs):
    global ENV
    if ENV is not None:
        ENV.close()
    ENV, description = ORIGINAL_ENV(*args, **kwargs)
    return ENV, description


class InfrastructureErrorHandler(logging.Handler):
    def emit(self, record):
        # Do not count an inference/transport exception as an ordinary task failure.
        if record.levelno >= logging.ERROR and record.getMessage().startswith("Caught exception:"):
            (OUT / "infrastructure-error.txt").write_text(record.getMessage())
            raise RuntimeError(record.getMessage())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger().addHandler(InfrastructureErrorHandler())
    upstream.imageio.mimwrite = recorded_video
    upstream._get_libero_env = managed_env
    (OUT / "protocol.json").write_text(json.dumps(vars(ARGS), indent=2))
    try:
        upstream.eval_libero(ARGS)
        assert len(RECORDS) == 100
        save_summary(completed=True)
        print("EVALUATION_COMPLETE", flush=True)
    finally:
        if ENV is not None:
            ENV.close()
