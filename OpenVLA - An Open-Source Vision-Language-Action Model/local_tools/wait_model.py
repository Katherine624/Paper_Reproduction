import json,sys,time
from pathlib import Path
r=Path("/root/gpufree-data/openvla-repro")
name=sys.argv[1]
rec=next(x for x in json.loads((r/"source_manifest.json").read_text()) if x["destination"]=="models/"+name)
while True:
    missing=[x["path"] for x in rec["files"] if not (r/rec["destination"]/x["path"]).exists() or (r/rec["destination"]/x["path"]).stat().st_size!=x["size"]]
    if not missing:print("Model ready:",name,flush=True);break
    print("Waiting for",name,len(missing),"files",flush=True);time.sleep(30)
