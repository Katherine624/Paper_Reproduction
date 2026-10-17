"""Export all locally produced OpenVLA artifacts, including optimizer checkpoints."""
import hashlib,io,json,os,subprocess,tarfile,time
from pathlib import Path
ROOT=Path("/root/gpufree-data/openvla-repro")
PREFIX="openvla_full_backup"
files={}
for folder in ["results","logs","scripts","openvla/rollouts"]:
    for p in (ROOT/folder).rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and p.name!="full-backup-export.log":
            assert not p.is_symlink(),str(p)
            files[p.relative_to(ROOT).as_posix()]=p
for name in ["env.sh","source_manifest.json","wheel_manifest.json","extra_wheel_manifest.json"]:
    files[name]=ROOT/name
# Include existing report and reference commands as historical experiment records.
for p in (ROOT/"delivery").glob("*"):
    if p.is_file() and p.name not in ["source_manifest.json","wheel_manifest.json","extra_wheel_manifest.json","env.sh"]:
        files["experiment_report/"+p.name]=p
records=[]
for name,p in sorted(files.items()):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    records.append({"path":name,"bytes":p.stat().st_size,"sha256":h.hexdigest()})
manifest={"created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"scope":"All generated results, checkpoints, scripts, logs and rollout videos; public base models, raw dataset and environments are excluded.","files":records,"file_count":len(records),"total_bytes":sum(x["bytes"] for x in records)}
(ROOT/"full_backup_manifest.json").write_text(json.dumps(manifest,indent=2))
output=ROOT/"openvla_full_backup.tar.gz"
partial=output.with_suffix(output.suffix+".part")
with tarfile.open(partial,"w:gz",compresslevel=1) as tar:
    for i,(name,p) in enumerate(sorted(files.items()),1):
        tar.add(p,arcname=PREFIX+"/"+name,recursive=False)
        if p.suffix in [".pt",".safetensors"]:print(json.dumps({"packed":name,"bytes":p.stat().st_size}),flush=True)
    b=json.dumps(manifest,indent=2).encode()
    info=tarfile.TarInfo(PREFIX+"/backup_manifest.json");info.size=len(b);info.mtime=int(time.time())
    tar.addfile(info,io.BytesIO(b))
os.replace(partial,output)
h=hashlib.sha256()
with output.open("rb") as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
summary={"status":"complete","archive":output.name,"archive_bytes":output.stat().st_size,"archive_sha256":h.hexdigest(),"file_count":len(records),"uncompressed_bytes":manifest["total_bytes"],"checkpoints":[50,100,105],"videos":sum(x["path"].endswith(".mp4") for x in records)}
(ROOT/"full_backup_export.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
