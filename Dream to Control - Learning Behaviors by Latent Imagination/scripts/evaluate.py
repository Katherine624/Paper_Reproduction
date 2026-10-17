"""Evaluate an exported checkpoint without constructing optimizers or replay."""
import argparse
import csv
import datetime
import json
import pathlib
import time
import common
from common import np, tf, tools, wrappers
import imageio.v2 as imageio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--video-episodes', type=int, default=1)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'episodes.jsonl').exists():
        raise FileExistsError('Choose a new output directory to avoid mixing evaluations')
    config = tools.AttrDict(json.loads((args.checkpoint / 'config.json').read_text()))
    common.configure(args.seed, config.precision)
    suite, task = config.task.split('_', 1)
    assert suite == 'dmc'
    probe_env = wrappers.DeepMindControl(task, seed=args.seed)
    agent = common.InferencePolicy(config, probe_env.action_space.shape[0])
    restore = agent.load(args.checkpoint)
    probe_env.close()
    before = {k: common.module_digest(m) for k, m in common.policy_modules(agent).items()}
    results = []
    started = time.monotonic()
    for episode in range(args.episodes):
        seed = args.seed + episode
        tf.random.set_seed(seed)
        base_env = wrappers.DeepMindControl(task, seed=seed)
        env = wrappers.NormalizeActions(wrappers.ActionRepeat(base_env, config.action_repeat))
        env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
        obs, state, total, decisions = env.reset(), None, 0.0, 0
        video = None
        if episode < args.video_episodes:
            fps = 1 / (base_env._env.control_timestep() * config.action_repeat)
            video = imageio.get_writer(args.output / f'episode_{episode:02d}.mp4', fps=fps)
        try:
            while True:
                action, state = agent.action(tf.convert_to_tensor(obs['image'][None]), state)
                action = action.numpy()[0]
                assert np.all(np.isfinite(action)) and np.all(np.abs(action) <= 1.00001)
                if video is not None:
                    video.append_data(base_env._env.physics.render(256, 256, camera_id=base_env._camera))
                obs, reward, done, _ = env.step(action)
                total += reward
                decisions += 1
                if done:
                    break
        finally:
            if video is not None:
                video.close()
            env.close()
        result = {'task': config.task, 'episode': episode, 'seed': seed,
                  'return': float(total), 'decisions': decisions,
                  'environment_steps': decisions * config.action_repeat}
        assert result['environment_steps'] == config.time_limit
        results.append(result)
        with (args.output / 'episodes.jsonl').open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        print('EVALUATION', json.dumps(result), flush=True)
    after = {k: common.module_digest(m) for k, m in common.policy_modules(agent).items()}
    assert before == after, 'Policy parameters changed during evaluation'
    returns = np.array([r['return'] for r in results])
    with (args.output / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {'status': 'completed', 'task': config.task, 'episodes': args.episodes,
               'mean_return': float(returns.mean()),
               'std_return': float(returns.std(ddof=1)) if len(returns) > 1 else None,
               'min_return': float(returns.min()), 'max_return': float(returns.max()),
               'elapsed_seconds': time.monotonic() - started,
               'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'weights_unchanged': before == after, 'parameter_sha256': after,
               'checkpoint_source': str(args.checkpoint.resolve()),
               'training_environment_steps': json.loads((args.checkpoint / 'training_audit.json').read_text())['environment_steps'],
               'evaluation_protocol': 'actor sampled-mode approximation; stochastic RSSM; no exploration noise; one independently reset environment per seed',
               'scope': 'short-training pipeline verification, not paper-level performance reproduction',
               **restore}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
