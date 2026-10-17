"""Cross-check delivered records and weights without importing TensorFlow."""
import argparse
import hashlib
import json
import math
import pathlib
import statistics


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', required=True, type=pathlib.Path)
    parser.add_argument('--evaluation', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    records = lambda path: [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    manifest = read(args.training / 'run_manifest.json')
    config = read(args.training / 'config.json')
    audit = read(args.training / 'training_audit.json')
    metadata = read(args.training / 'policy_weights.json')
    summary = read(args.evaluation / 'summary.json')
    episodes = records(args.evaluation / 'episodes.jsonl')
    metrics = records(args.training / 'metrics.jsonl')
    assert manifest['status'] == summary['status'] == 'completed'
    assert audit['environment_steps'] == config['steps'] == summary['training_environment_steps']
    assert all(audit['modules_changed'].values())
    assert all(x > 0 for x in audit['optimizer_iterations'].values())
    for relative, digest in manifest['source_sha256'].items():
        assert sha256(root / relative) == digest, relative
    assert sha256(args.training / 'variables.pkl') == audit['checkpoint_sha256']
    assert sha256(args.training / 'policy_weights.npz') == audit['policy_sha256'] == metadata['sha256'] == summary['checkpoint_sha256']
    assert summary['weights_unchanged'] and summary['restore_max_abs_difference'] == 0
    for key, digest in summary['parameter_sha256'].items():
        assert audit['current_module_sha256'][key] == digest, key
    assert len(episodes) == summary['episodes']
    assert len({r['seed'] for r in episodes}) == len(episodes)
    assert all(r['environment_steps'] == config['time_limit'] for r in episodes)
    assert all(math.isfinite(r['return']) for r in episodes)
    returns = [r['return'] for r in episodes]
    assert math.isclose(statistics.mean(returns), summary['mean_return'], abs_tol=1e-9)
    if len(episodes) > 1:
        assert math.isclose(statistics.stdev(returns), summary['std_return'], abs_tol=1e-9)
    loss_rows = [r for r in metrics if 'model_loss' in r]
    assert loss_rows
    for row in loss_rows:
        for key in ('model_loss', 'actor_loss', 'value_loss', 'model_grad_norm', 'actor_grad_norm', 'value_grad_norm'):
            assert math.isfinite(row[key]), (row['step'], key)
    result = {
        'status': 'passed', 'training_status': manifest['status'],
        'environment_steps': audit['environment_steps'],
        'source_hashes_match_run': True, 'checkpoint_hashes_match': True,
        'all_six_model_components_changed': True,
        'optimizer_iterations': audit['optimizer_iterations'],
        'finite_loss_and_gradient_rows': len(loss_rows),
        'evaluation_episodes': len(episodes), 'unique_evaluation_seeds': len(episodes),
        'evaluation_weights_unchanged': True, 'restore_max_abs_difference': 0.0,
        'episode_statistics_recomputed': True,
        'mean_return': summary['mean_return'], 'std_return': summary['std_return'],
        'scope': 'Short-run pipeline verification; no claim of paper-level performance.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
