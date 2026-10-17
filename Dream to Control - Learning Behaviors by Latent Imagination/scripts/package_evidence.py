"""Archive a completed experiment, including replay, without bundling runtimes."""
import argparse
import hashlib
import json
import pathlib
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('output/dreamer_v1_evidence.tar.gz'))
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    verification = json.loads((root / 'output/verification.json').read_text())
    assert verification['status'] == 'passed'
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    included = []
    paths = ['input', 'src', 'scripts', 'patches', 'runs/smoke', 'runs/walker_walk_seed0',
             'output/runtime_probe', 'output/smoke_eval', 'output/walker_walk_eval10',
             'output/figures', 'output/core_validation.json', 'output/verification.json',
             'output/system_info.json', 'output/reproduction_report.md', 'README.md', 'requirements.in',
             'requirements.lock.txt', 'runtime_sources.json', 'reproduction_spec.json',
             'logs/runtime_probe_tf220_failed.log', 'logs/preflight_and_smoke.log',
             'logs/smoke_resume_eval.log', 'logs/walker_walk_short.log']
    files = []
    for relative in paths:
        path = root / relative
        assert path.exists(), relative
        files.extend(sorted(path.rglob('*')) if path.is_dir() else [path])
    with tarfile.open(destination, 'w:gz', compresslevel=3) as archive:
        for path in files:
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            included.append({'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            archive.add(path, arcname=relative, recursive=False)
    manifest = {'archive': destination.name, 'bytes': destination.stat().st_size,
                'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                'includes_replay': True, 'includes_python_environment': False,
                'files': included}
    destination.with_suffix('').with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}))


if __name__ == '__main__':
    main()
