"""Verify the downloaded archive and every exported file before marking backup complete."""
import datetime,hashlib,json,tarfile
from pathlib import Path
OUTPUT=Path(__file__).resolve().parents[1]/"output"
ARCHIVE=OUTPUT/"openvla_full_backup.tar.gz"
ROOT=OUTPUT/"openvla_full_backup"
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
expected=json.loads((OUTPUT/"full_backup_export.json").read_text())
assert ARCHIVE.stat().st_size==expected["archive_bytes"],"Archive is incomplete"
digest=sha(ARCHIVE)
assert digest==expected["archive_sha256"],"Archive SHA256 mismatch"
print("Archive SHA256 verified; extracting.",flush=True)
root_resolved=ROOT.resolve()
with tarfile.open(ARCHIVE,"r|gz") as tar:
    for m in tar:
        dest=(OUTPUT/m.name).resolve()
        assert dest.is_relative_to(root_resolved) and not m.issym() and not m.islnk(),m.name
        assert m.isfile() or m.isdir(),m.name
        tar.extract(m,OUTPUT,filter="data")
manifest=json.loads((ROOT/"backup_manifest.json").read_text())
external=json.loads((OUTPUT/"full_backup_manifest.json").read_text())
assert manifest==external
for record in manifest["files"]:
    p=ROOT/record["path"]
    assert p.is_file() and p.stat().st_size==record["bytes"],record["path"]
    assert sha(p)==record["sha256"],record["path"]
    if p.suffix in [".pt",".safetensors"]:print("Verified "+record["path"],flush=True)
videos=list((ROOT/"openvla/rollouts").rglob("*.mp4"))
assert len(videos)==expected["videos"]==103
for step in [50,100,105]:
    cp=ROOT/f"results/train-smoke/checkpoint-{step}"
    for name in ["adapter_model.safetensors","training_state.pt","adapter_config.json","dataset_statistics.json","tokenizer.json"]:
        assert (cp/name).is_file()
train=ROOT/"results/train-smoke"
assert [json.loads(x)["step"] for x in (train/"metrics.jsonl").read_text().splitlines()]==list(range(1,101))
assert [json.loads(x)["step"] for x in (train/"resume_metrics.jsonl").read_text().splitlines()]==list(range(101,106))
result={"status":"passed","verified_at":datetime.datetime.now().astimezone().isoformat(),"archive":str(ARCHIVE),"archive_sha256":digest,"verified_files":len(manifest["files"]),"verified_bytes":sum(x["bytes"] for x in manifest["files"]),"checkpoints":[50,100,105],"videos":len(videos),"root":str(ROOT)}
(ROOT/"local_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False),flush=True)
