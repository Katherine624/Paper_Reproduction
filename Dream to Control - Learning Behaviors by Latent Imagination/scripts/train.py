"""Run the author's Dreamer v1 with explicit manifests and policy exports."""
import argparse
import datetime
import json
import pathlib
import sys
import time
import traceback
import common
from common import dreamer, tf

class AuditedDreamer(dreamer.Dreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_digests = self.digests()

    def digests(self):
        return {name: common.module_digest(getattr(self, '_' + name))
                for name in ('encode', 'dynamics', 'decode', 'reward', 'actor', 'value')}

    def load(self, filename):
        super().load(filename)
        self.initial_digests = self.digests()
        print('CHECKPOINT_RESTORED', str(filename), 'environment_steps',
              int(self._step.numpy()), flush=True)

    def save(self, filename):
        super().save(filename)
        metadata = common.export_policy(self, pathlib.Path(filename).parent)
        current = self.digests()
        record = {'environment_steps': int(self._step.numpy()),
                  'initial_module_sha256': dict(self.initial_digests),
                  'current_module_sha256': current,
                  'modules_changed': {k: self.initial_digests[k] != v for k, v in current.items()},
                  'optimizer_iterations': {name: int(getattr(self, '_' + name + '_opt')._opt.iterations.numpy())
                                           for name in ('model', 'actor', 'value')},
                  'checkpoint_sha256': common.sha256(filename),
                  'policy_sha256': metadata['sha256']}
        (pathlib.Path(filename).parent / 'training_audit.json').write_text(json.dumps(record, indent=2))
        print('CHECKPOINT_AUDIT', json.dumps(record), flush=True)

def main():
    parser = argparse.ArgumentParser()
    for key, value in dreamer.define_config().items():
        parser.add_argument('--' + key, type=common.tools.args_type(value), default=value)
    config = parser.parse_args()
    common.configure(config.seed, config.precision)
    config.logdir.mkdir(parents=True, exist_ok=True)
    path = config.logdir / 'run_manifest.json'
    serializable = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(config).items()}
    config_path = config.logdir / 'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        for key in ('task', 'seed', 'cnn_depth', 'num_units', 'stoch_size', 'deter_size',
                    'action_repeat', 'batch_size', 'batch_length', 'precision'):
            if previous[key] != serializable[key]:
                raise ValueError(f'Resume config mismatch: {key}')
    config_path.write_text(json.dumps(serializable, indent=2))
    manifest = {'upstream': 'https://github.com/danijar/dreamer',
                'upstream_commit': common.UPSTREAM_COMMIT,
                'status': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'command': sys.argv, 'working_directory': str(pathlib.Path.cwd()),
                'config': serializable, 'tensorflow': tf.__version__,
                'source_sha256': {str(p.relative_to(common.ROOT)): common.sha256(p)
                                  for directory in ('src/dreamer', 'scripts')
                                  for p in (common.ROOT / directory).glob('*.py')}}
    path.write_text(json.dumps(manifest, indent=2))
    with (config.logdir / 'run_history.jsonl').open('a') as stream:
        stream.write(json.dumps(manifest) + '\n')
    dreamer.Dreamer = AuditedDreamer
    start = time.monotonic()
    try:
        dreamer.main(config)
        manifest['status'] = 'completed'
    except BaseException:
        manifest['status'] = 'failed'
        manifest['error'] = traceback.format_exc()
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - start
        manifest['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        path.write_text(json.dumps(manifest, indent=2))
        with (config.logdir / 'run_history.jsonl').open('a') as stream:
            stream.write(json.dumps(manifest) + '\n')

if __name__ == '__main__':
    main()
