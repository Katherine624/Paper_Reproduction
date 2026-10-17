"""Restore locally trained checkpoint shards from this repository's release assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "checkpoint_manifest.json"


def restore_file(file_info: dict, release_base_url: str) -> None:
    destination = ROOT / file_info["path"]
    if destination.is_file() and destination.stat().st_size == file_info["size"]:
        with destination.open("rb") as existing:
            existing_hash = hashlib.file_digest(existing, "sha256").hexdigest()
        if existing_hash == file_info["sha256"]:
            print(f"Already verified: {destination}")
            return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    whole_hash = hashlib.sha256()
    with partial.open("wb") as output:
        for chunk in file_info["chunks"]:
            url = release_base_url.rstrip("/") + "/" + chunk["name"]
            request = Request(url, headers={"User-Agent": "tau0-wm-checkpoint-restore"})
            chunk_hash = hashlib.sha256()
            count = 0
            print(f"Downloading {chunk['name']} ...", flush=True)
            with urlopen(request, timeout=120) as response:
                while block := response.read(8 * 1024 * 1024):
                    output.write(block)
                    whole_hash.update(block)
                    chunk_hash.update(block)
                    count += len(block)
            if count != chunk["size"] or chunk_hash.hexdigest() != chunk["sha256"]:
                raise ValueError(f"Chunk verification failed: {chunk['name']}")

    if partial.stat().st_size != file_info["size"] or whole_hash.hexdigest() != file_info["sha256"]:
        raise ValueError(f"Checkpoint verification failed: {destination}")
    partial.replace(destination)
    print(f"Restored and verified: {destination}")


def main() -> None:
    with MANIFEST.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for file_info in manifest["files"]:
        restore_file(file_info, manifest["release_base_url"])


if __name__ == "__main__":
    main()
