"""Download a pinned, real LIBERO subset for a training smoke run."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
REPO = "physical-intelligence/libero"
EPISODES = list(range(8))
DEST = ROOT / "data" / "libero-subset"
FILES = ["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl", "meta/stats.json"]
FILES += [f"data/chunk-000/episode_{i:06d}.parquet" for i in EPISODES]


def download(relative):
    path = DEST / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{relative}?download=true"
        part = path.with_name(path.name + ".part")
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=90) as response, part.open("wb") as out:
                    expected = response.headers.get("Content-Length")
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                if expected and part.stat().st_size != int(expected):
                    raise ValueError("incomplete download")
                part.replace(path)
                break
            except Exception as exc:
                print("RETRY", relative, attempt + 1, type(exc).__name__, flush=True)
                if attempt == 5:
                    raise
                time.sleep(min(attempt + 1, 5))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print("DONE", relative, path.stat().st_size, flush=True)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": digest}


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(download, FILES))
    manifest = {"source_repo": REPO, "revision": REVISION, "episodes": EPISODES,
                "purpose": "real-data short training verification; not converged training",
                "metadata": "upstream full metadata retained; loader must explicitly select episodes",
                "files": files}
    (DEST / "subset-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = ROOT / "work" / "remote-setup" / "libero-subset.tar"
    with tarfile.open(archive, "w") as tar:
        for relative in FILES + ["subset-manifest.json"]:
            tar.add(DEST / relative, arcname=relative)
    print("READY", str(archive), archive.stat().st_size, flush=True)
