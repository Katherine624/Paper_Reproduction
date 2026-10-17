import concurrent.futures,hashlib,json,pathlib,time,urllib.request
ROOT=pathlib.Path("/root/gpufree-data/openvla-repro")
records=json.loads((ROOT/"extra_wheel_manifest.json").read_text())
CHUNK=4*1024*1024
def part(rec,i):
    start=i*CHUNK;end=min(rec["size"]-1,start+CHUNK-1)
    p=ROOT/"cache/wheel-parts"/rec["filename"]/str(i);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.stat().st_size==end-start+1:return
    for attempt in range(6):
        try:
            req=urllib.request.Request(rec["url"]+f"?segment={i}&attempt={attempt}",headers={"Range":f"bytes={start}-{end}"})
            with urllib.request.urlopen(req,timeout=30) as r:
                cr=r.headers.get("Content-Range","");b=r.read(end-start+2)
            assert cr.startswith(f"bytes {start}-") and len(b)==end-start+1,(cr,len(b))
            p.write_bytes(b);return
        except Exception as e:
            if attempt==5:raise
            time.sleep(1+attempt)
def assemble(rec):
    dst=ROOT/"wheels"/rec["filename"];h=hashlib.sha256()
    with dst.open("wb") as out:
        for i in range((rec["size"]+CHUNK-1)//CHUNK):
            b=(ROOT/"cache/wheel-parts"/rec["filename"]/str(i)).read_bytes();out.write(b);h.update(b)
    assert h.hexdigest()==rec["sha256"],rec["filename"]
    print(json.dumps({"event":"wheel_complete","file":rec["filename"],"size":rec["size"]}),flush=True)
start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
    futures={};left={}
    for r in records:
        dst=ROOT/"wheels"/r["filename"]
        if dst.exists() and dst.stat().st_size==r["size"] and hashlib.sha256(dst.read_bytes()).hexdigest()==r["sha256"]:continue
        n=(r["size"]+CHUNK-1)//CHUNK;left[r["filename"]]=n
        for i in range(n):futures[pool.submit(part,r,i)]=r
    completed=0
    for f in concurrent.futures.as_completed(futures):
        f.result();r=futures[f];left[r["filename"]]-=1;completed+=1
        if completed%50==0:print(json.dumps({"chunks":completed,"total":len(futures),"seconds":round(time.monotonic()-start)}),flush=True)
        if left[r["filename"]]==0:assemble(r)
(ROOT/"results/extra_wheels_complete.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
