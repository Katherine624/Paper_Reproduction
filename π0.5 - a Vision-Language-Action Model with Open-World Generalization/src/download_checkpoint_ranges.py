"""Concurrent HTTP ranges with resumable chunks and final official MD5 checks."""
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/python-tools"))
import google_crc32c
DEST = ROOT / "data/official-checkpoints"
CHUNK = 32 * 1024 * 1024


def valid(path, item):
    if not path.exists() or path.stat().st_size != int(item["size"]):
        return False
    with path.open("rb") as f:
        if "md5Hash" in item:
            digest = hashlib.file_digest(f, "md5").digest()
            expected = item["md5Hash"]
        else:
            crc = google_crc32c.Checksum()
            while block := f.read(8 * 1024 * 1024):
                crc.update(block)
            digest = crc.digest()
            expected = item["crc32c"]
        return base64.b64encode(digest).decode() == expected


def fetch_chunk(job):
    item, start, end, dest = job
    if dest.exists() and dest.stat().st_size == end - start + 1:
        return dest.stat().st_size
    url = "https://storage.googleapis.com/openpi-assets/" + item["name"]
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=90) as r:
                assert r.status == 206, r.status
                assert r.headers["Content-Range"].startswith(f"bytes {start}-{end}/")
                data = r.read(end - start + 2)
            assert len(data) == end - start + 1
            temp = dest.with_suffix(".incomplete")
            temp.write_bytes(data)
            temp.replace(dest)
            return len(data)
        except Exception as exc:
            print("RETRY", item["name"].split("/")[-1], start, attempt + 1, type(exc).__name__, flush=True)
            if attempt == 5:
                raise
            time.sleep(min(attempt + 1, 5))


if __name__ == "__main__":
    for model in ["pi05_base", "pi05_libero"]:
        manifest_path = ROOT / "work/remote-setup" / f"{model}-asset-list.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        else:
            url = f"https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/{model}/&maxResults=1000"
            with urllib.request.urlopen(url, timeout=30) as r:
                manifest = json.load(r)
            assert not manifest.get("nextPageToken")
            manifest_path.write_text(json.dumps(manifest, indent=2))
        tasks, plans = [], []
        present = 0
        for item in manifest["items"]:
            path = DEST / model / item["name"].removeprefix(f"checkpoints/{model}/")
            path.parent.mkdir(parents=True, exist_ok=True)
            if valid(path, item):
                present += int(item["size"])
                continue
            part = path.with_name(path.name + ".part")
            part.touch(exist_ok=True)
            start = part.stat().st_size
            present += start
            chunkdir = ROOT / "work/checkpoint-chunks" / model / path.name
            chunkdir.mkdir(parents=True, exist_ok=True)
            chunks = []
            for offset in range(start, int(item["size"]), CHUNK):
                end = min(offset + CHUNK, int(item["size"])) - 1
                chunk = chunkdir / f"{offset:012d}-{end:012d}.chunk"
                tasks.append((item, offset, end, chunk))
                chunks.append(chunk)
            plans.append((item, path, part, chunks))
        print("START", model, "existing_GB", round(present / 1e9, 3), "range_jobs", len(tasks), flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            pending = {pool.submit(fetch_chunk, task) for task in tasks}
            while pending:
                done, pending = concurrent.futures.wait(pending, timeout=30, return_when=concurrent.futures.FIRST_EXCEPTION)
                for f in done:
                    present += f.result()
                print("PROGRESS", model, round(present / 1e9, 3), "GB", "remaining_ranges", len(pending), flush=True)
        for item, path, part, chunks in plans:
            with part.open("ab") as out:
                for chunk in chunks:
                    with chunk.open("rb") as source:
                        shutil.copyfileobj(source, out, 1024 * 1024)
                    out.flush()
                    chunk.unlink()
            if not valid(part, item):
                raise ValueError("Official checksum mismatch: " + str(path))
            part.replace(path)
            print("VERIFIED", model, path.name, flush=True)
        (DEST / model / "download-verified.json").write_text(json.dumps(manifest, indent=2))
        print("MODEL_READY", model, flush=True)
