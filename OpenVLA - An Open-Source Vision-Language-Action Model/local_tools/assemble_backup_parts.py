"""Reassemble native-SFTP parts only after both transfer sessions complete."""
import hashlib,json,os
from pathlib import Path
root=Path(__file__).resolve().parents[1]/"output"
summary=json.loads((root/"full_backup_export.json").read_text())
parts=[root/"openvla_full_backup.tar.gz",root/"openvla_backup_part-01",root/"openvla_backup_part-02"]
sizes=[1073741824,1073741824,summary["archive_bytes"]-2147483648]
for p,n in zip(parts,sizes):assert p.stat().st_size==n,(p.name,p.stat().st_size,n)
target=root/"openvla_full_backup.assembled.tar.gz"
h=hashlib.sha256()
with target.open("wb") as f:
    for p in parts:
        with p.open("rb") as src:
            for b in iter(lambda:src.read(8*1024*1024),b""):f.write(b);h.update(b)
assert h.hexdigest()==summary["archive_sha256"],"Reassembled archive checksum mismatch"
os.replace(target,parts[0])
print(json.dumps({"status":"assembled","bytes":parts[0].stat().st_size,"sha256":h.hexdigest()}),flush=True)
