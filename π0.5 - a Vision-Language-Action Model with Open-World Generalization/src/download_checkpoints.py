"""Relay official checkpoints through the local network; verify GCS MD5."""
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "official-checkpoints"
CHUNK = 32 * 1024 * 1024
PRINT_LOCK = threading.Lock()


def log(*args):
    with PRINT_LOCK:
        print(*args, flush=True)


def fetch(job):
    model, item = job
    relative = item["name"].removeprefix(f"checkpoints/{model}/")
    path = DEST / model / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    size = int(item["size"])

    def valid(p):
        if not p.exists() or p.stat().st_size != size:
            return False
        with p.open("rb") as f:
            return base64.b64encode(hashlib.file_digest(f, "md5").digest()).decode() == item["md5Hash"]

    if valid(path):
        return
    part = path.with_name(path.name + ".part")
    part.touch(exist_ok=True)
    url = "https://storage.googleapis.com/openpi-assets/" + item["name"]
    while part.stat().st_size < size:
        start = part.stat().st_size
        end = min(size, start + CHUNK) - 1
        for attempt in range(6):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206:
                        raise ValueError(f"Range request returned {response.status}")
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError("incomplete range")
                with part.open("ab") as f:
                    f.write(data)
                break
            except Exception as exc:
                log("RETRY", model, relative, start, attempt + 1, type(exc).__name__)
                if attempt == 5:
                    raise
                time.sleep(min(2 * (attempt + 1), 10))
    if not valid(part):
        raise ValueError(f"MD5 mismatch: {model}/{relative}")
    part.replace(path)
    log("VERIFIED", model, relative, size)


if __name__ == "__main__":
    for model in ["pi05_base", "pi05_libero"]:
        url = f"https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/{model}/&maxResults=1000"
        with urllib.request.urlopen(url, timeout=30) as response:
            manifest = json.load(response)
        assert not manifest.get("nextPageToken")
        (ROOT / "work" / "remote-setup" / f"{model}-asset-list.json").write_text(json.dumps(manifest, indent=2))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            pending = {pool.submit(fetch, (model, item)) for item in manifest["items"]}
            while pending:
                done, pending = concurrent.futures.wait(pending, timeout=30, return_when=concurrent.futures.FIRST_EXCEPTION)
                for future in done:
                    future.result()
                actual = sum(p.stat().st_size for p in (DEST / model).rglob("*") if p.is_file())
                log("PROGRESS", model, round(actual / 1e9, 3), "GB", "remaining_files", len(pending))
        (DEST / model / "download-verified.json").write_text(json.dumps({"source": "gs://openpi-assets/checkpoints/" + model, "files": manifest["items"]}, indent=2))
        log("MODEL_READY", model)
