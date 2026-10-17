#!/usr/bin/env python3
"""Download, verify, and restore the public Cosmos Policy Release backup in Linux/WSL."""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys, time, urllib.request

def download(part, folder):
    target = folder / part['name']
    for attempt in range(6):
        try:
            if target.exists():
                with target.open('rb') as f:
                    h = hashlib.sha256()
                    while block := f.read(4 * 1024 * 1024): h.update(block)
                if target.stat().st_size == part['size'] and h.hexdigest() == part['sha256']:
                    return target
            temp = target.with_suffix(target.suffix + '.download')
            h = hashlib.sha256(); size = 0
            request = urllib.request.Request(part['url'], headers={'User-Agent': 'Cosmos-Backup-Restore'})
            with urllib.request.urlopen(request, timeout=120) as response, temp.open('wb') as f:
                while block := response.read(4 * 1024 * 1024):
                    f.write(block); h.update(block); size += len(block)
            if size != part['size'] or h.hexdigest() != part['sha256']:
                raise RuntimeError('SHA-256 or size mismatch: ' + part['name'])
            os.replace(temp, target)
            return target
        except (OSError, RuntimeError):
            if attempt == 5: raise
            time.sleep(min(30, 2 ** attempt))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=pathlib.Path, required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True, help='A NEW empty directory; archives restore into project/ and linux-root/')
    ap.add_argument('--groups', nargs='+', help='Defaults to every archive; e.g. project models')
    ap.add_argument('--resume', action='store_true', help='Resume a directory previously created by this script')
    args = ap.parse_args()
    if not sys.platform.startswith('linux') or not shutil.which('tar'):
        ap.error('Run with Python 3.10+ in Linux/WSL, with GNU tar installed.')
    m = json.loads(args.manifest.read_text(encoding='utf-8'))
    if not m.get('complete'): ap.error('The backup manifest is incomplete.')
    names = args.groups or list(m['archives'])
    if any(n not in m['archives'] for n in names): ap.error('Unknown archive group.')
    out = args.output.expanduser().resolve()
    marker = out / '.cosmos-restore-state.json'
    if out.exists() and any(out.iterdir()):
        if not args.resume or not marker.is_file(): ap.error('Output must be empty, or use --resume with an existing restore directory.')
        status = json.loads(marker.read_text())
        if status['tag'] != m['tag']: ap.error('Output belongs to a different backup.')
    else:
        out.mkdir(parents=True, exist_ok=True)
        status = {'tag':m['tag'], 'completed':[]}
        marker.write_text(json.dumps(status, indent=2))
    chunks = out / '.parts'; chunks.mkdir(exist_ok=True)
    for name in names:
        if name in status['completed']:
            print('Already restored:', name, flush=True); continue
        group = m['archives'][name]
        if not group['complete']: raise RuntimeError('Incomplete archive: ' + name)
        dest = out / ('linux-root' if name == 'linux-runtime' else 'project')
        dest.mkdir(exist_ok=True)
        parts = sorted(group['parts'].values(), key=lambda p:p['index'])
        if [p['index'] for p in parts] != list(range(1, group['part_count'] + 1)):
            raise RuntimeError('Part sequence is incomplete: ' + name)
        whole = hashlib.sha256()
        proc = subprocess.Popen(['tar', '--no-same-owner', '-xzf', '-', '-C', str(dest)], stdin=subprocess.PIPE)
        try:
            for part in parts:
                p = download(part, chunks)
                print('Verified:', part['name'], flush=True)
                with p.open('rb') as f:
                    while block := f.read(4 * 1024 * 1024):
                        whole.update(block); proc.stdin.write(block)
                p.unlink()  # Only a downloaded, verified temporary part is removed.
            proc.stdin.close()
            if proc.wait() or whole.hexdigest() != group['sha256']:
                raise RuntimeError('Archive extraction or whole-archive hash failed: ' + name)
        except BaseException:
            proc.kill(); proc.wait(); raise
        status['completed'].append(name)
        marker.write_text(json.dumps(status, indent=2))
        print('Restored:', name, 'to', dest, flush=True)
    print('Restore complete. Read RESTORE.md before activating the archived environment.')

if __name__ == '__main__':
    main()
