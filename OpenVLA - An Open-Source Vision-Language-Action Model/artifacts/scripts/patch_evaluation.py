"""Add bounded evaluation and machine-readable evidence without changing policy behavior."""
from pathlib import Path
import os
ROOT=Path(os.environ.get("OPENVLA_REPO", "/root/gpufree-data/openvla-repro/openvla"))
p=ROOT/"experiments/robot/libero/run_libero_eval.py"
s=p.read_text()
def replace(old,new):
    global s
    if s.count(old)!=1: raise RuntimeError("Patch anchor mismatch: "+old[:80])
    s=s.replace(old,new)
replace("import os\n", "import os\nimport json\nimport time\nimport torch\n")
replace('    num_trials_per_task: int = 50', '    task_ids: Optional[str] = None                  # Optional comma-separated task ids for smoke runs\n    num_trials_per_task: int = 50')
replace('    log_file = open(local_log_filepath, "w")', '    log_file = open(local_log_filepath, "w")\n    evidence = open(os.path.join(cfg.local_log_dir, run_id + ".jsonl"), "w", buffering=1)\n    with open(os.path.join(cfg.local_log_dir, run_id + ".config.json"), "w") as f:\n        json.dump(vars(cfg), f, indent=2, default=str)\n    action_latencies = []\n    torch.cuda.reset_peak_memory_stats()')
replace('    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):', '    selected_ids = list(range(num_tasks_in_suite)) if cfg.task_ids is None else [int(x) for x in cfg.task_ids.split(",")]\n    assert all(0 <= x < num_tasks_in_suite for x in selected_ids)\n    for task_id in tqdm.tqdm(selected_ids):')
replace('            t = 0\n            replay_images = []', '            t = 0\n            done = False\n            episode_error = None\n            episode_start = time.perf_counter()\n            replay_images = []')
replace('                    action = get_action(', '                    torch.cuda.synchronize()\n                    action_start = time.perf_counter()\n                    action = get_action(')
replace('                    # Normalize gripper action [0,1]', '                    torch.cuda.synchronize()\n                    action_latencies.append(time.perf_counter() - action_start)\n                    assert np.asarray(action).shape == (7,) and np.isfinite(action).all(), "Invalid action"\n\n                    # Normalize gripper action [0,1]')
replace('                    print(f"Caught exception: {e}")', '                    episode_error = repr(e)\n                    done = False\n                    print(f"Caught exception: {e}")')
replace('            # Save a replay video of the episode', '            evidence.write(json.dumps({"task_id": task_id, "task": task_description, "seed": cfg.seed,\n                "initial_state_id": episode_idx, "success": bool(done), "steps": t,\n                "seconds": time.perf_counter() - episode_start, "error": episode_error}) + "\\n")\n\n            # Save a replay video of the episode')
replace('        # Log final results', '        env.close()\n\n        # Log final results')
replace('    # Save local log file', '    timing = np.asarray(action_latencies[10:])\n    summary = {"episodes": total_episodes, "successes": total_successes,\n        "success_rate": total_successes / total_episodes, "seed": cfg.seed,\n        "action_latency_p50_s": float(np.median(timing)) if timing.size else None,\n        "action_latency_p95_s": float(np.percentile(timing, 95)) if timing.size else None,\n        "timed_actions": int(timing.size), "warmup_actions_excluded": min(10, len(action_latencies)),\n        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()}\n    with open(os.path.join(cfg.local_log_dir, run_id + ".summary.json"), "w") as f:\n        json.dump(summary, f, indent=2)\n    evidence.close()\n    print(json.dumps(summary), flush=True)\n\n    # Save local log file')
p.write_text(s)
p=ROOT/"experiments/robot/openvla_utils.py"
s=p.read_text().replace("import tensorflow as tf\n","import tensorflow as tf\ntf.config.set_visible_devices([], \"GPU\")\n").replace("trust_remote_code=True", "trust_remote_code=False")
p.write_text(s)
print("Evaluation evidence patch applied")
