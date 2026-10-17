"""Download and verify the exact OpenVLA experiment archive using the stdlib."""
import argparse
import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('restored_backup'))
    parser.add_argument('--verify-only', action='store_true', help='Stream and verify; do not save any downloaded bytes.')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('release_manifest.json'))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    output = args.output.resolve()
    archive = output / manifest['archive_name']
    root = output / 'openvla_full_backup'
    partial = archive.with_name(archive.name + '.partial')
    if not args.verify_only:
        output.mkdir(parents=True, exist_ok=True)
        if archive.exists() or partial.exists() or root.exists():
            raise FileExistsError('Choose a new output directory to avoid overwriting existing backups.')
    whole = hashlib.sha256()
    total = 0
    sink = None if args.verify_only else partial.open('xb')
    try:
        for part in manifest['parts']:
            h = hashlib.sha256()
            count = 0
            last = time.monotonic()
            request = urllib.request.Request(part['url'], headers={'User-Agent': 'OpenVLA-reproduction-restore'})
            with urllib.request.urlopen(request, timeout=120) as response:
                for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                    count += len(block)
                    total += len(block)
                    h.update(block)
                    whole.update(block)
                    if sink:
                        sink.write(block)
                    if time.monotonic() - last > 20:
                        print(f"{part['name']}: {count / part['bytes']:.0%}", flush=True)
                        last = time.monotonic()
            if count != part['bytes'] or h.hexdigest() != part['sha256']:
                raise ValueError('Part failed size or SHA256 verification: ' + part['name'])
            print('Verified ' + part['name'], flush=True)
        if total != manifest['archive_bytes'] or whole.hexdigest() != manifest['archive_sha256']:
            raise ValueError('Combined archive verification failed')
    finally:
        if sink:
            sink.close()
    if args.verify_only:
        print('All remote parts and the combined archive SHA256 passed. No download was retained.', flush=True)
        return
    partial.rename(archive)
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            dest = (output / member.name).resolve()
            if not dest.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive member: ' + member.name)
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dest.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                dest.chmod(member.mode & 0o777)
    index = json.loads((root / 'backup_manifest.json').read_text(encoding='utf-8'))
    for item in index['files']:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root) or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError('Extracted file verification failed: ' + item['path'])
    if len(index['files']) != manifest['files']:
        raise ValueError('File-count mismatch')
    print(f"Restored and verified {len(index['files'])} files in {root}", flush=True)


if __name__ == '__main__':
    main()
