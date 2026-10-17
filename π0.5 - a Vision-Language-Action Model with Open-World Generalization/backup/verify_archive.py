import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
rows = json.loads((root / 'backup/archive-manifest.json').read_text(encoding='utf-8'))
for row in rows:
    path = root / row['path']
    assert path.stat().st_size == row['bytes'], row['path']
    with path.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == row['sha256'], row['path']
print(f'PASS: {len(rows)} files match the archive manifest.')
