import concurrent.futures, hashlib, json, os, pathlib, time, urllib.request
ROOT=pathlib.Path("/root/gpufree-data/openvla-repro")
MANIFEST=json.loads((ROOT/"source_manifest.json").read_text())
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while chunk:=f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()
def get(repo,entry):
    path=ROOT/repo["destination"]/entry["path"]
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+".part")
    if path.exists() and path.stat().st_size==entry["size"]:
        if not entry["sha256"] or digest(path)==entry["sha256"]:
            print(json.dumps({"event":"cached","file":str(path.relative_to(ROOT))}),flush=True)
            return
    prefix="datasets/" if repo["kind"]=="dataset" else ""
    url=f'https://hf-mirror.com/{prefix}{repo["repo"]}/resolve/{repo["revision"]}/{entry["path"]}'
    for attempt in range(8):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            headers={"User-Agent":"OpenVLA-Reproduction/1.0"}
            if offset: headers["Range"]=f"bytes={offset}-"
            req=urllib.request.Request(url+f"?download=true&attempt={attempt}",headers=headers)
            start=time.monotonic()
            with urllib.request.urlopen(req,timeout=60) as response:
                if offset and response.status!=206: offset=0
                with partial.open("ab" if offset else "wb") as out:
                    report=offset
                    while chunk:=response.read(4*1024*1024):
                        out.write(chunk);offset+=len(chunk)
                        if offset-report>=512*1024*1024:
                            print(json.dumps({"event":"progress","file":entry["path"],"bytes":offset,"total":entry["size"]}),flush=True)
                            report=offset
            if partial.stat().st_size!=entry["size"]:
                raise RuntimeError(f'size {partial.stat().st_size} != {entry["size"]}')
            actual=digest(partial)
            if entry["sha256"] and actual!=entry["sha256"]:
                partial.rename(partial.with_name(partial.name+f".bad-{int(time.time())}"))
                raise RuntimeError("sha256 mismatch")
            partial.replace(path)
            print(json.dumps({"event":"complete","file":str(path.relative_to(ROOT)),"bytes":entry["size"],"sha256":actual,"seconds":round(time.monotonic()-start,2)}),flush=True)
            return
        except Exception as e:
            print(json.dumps({"event":"retry","file":entry["path"],"attempt":attempt,"error":str(e)}),flush=True)
            time.sleep(min(2**attempt,15))
    raise RuntimeError("Download failed: "+entry["path"])
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(get,repo,entry) for repo in MANIFEST for entry in repo["files"]]
    for f in concurrent.futures.as_completed(futures):f.result()
(ROOT/"results/download_complete.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
