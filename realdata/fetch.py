"""Download the images listed in a realdata manifest (clean_frames.csv or real_glitches.csv).

    python realdata/fetch.py realdata/clean_frames.csv data/real_frames

Skips files already present, so it is safe to re-run. Rows whose download fails are
reported and skipped (the loaders only use files that exist).
"""

from __future__ import annotations

import csv
import sys
import time
import urllib.request
from pathlib import Path

UA = "gpu-corruptnet-research/0.1 (https://github.com/tanaymihani/gpu-corruptnet)"


def fetch(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                dest.write_bytes(r.read())
            return True
        except Exception as e:  # noqa: BLE001 - retry transient errors, then give up
            if getattr(e, "code", None) == 404:
                return False
            time.sleep(3 * (attempt + 1))
    return False


def main(manifest: str, root: str) -> None:
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8")))
    failed = 0
    for i, row in enumerate(rows):
        dest = out / row["file"]
        if dest.exists() and dest.stat().st_size > 0:
            continue
        if not fetch(row["url"], dest):
            failed += 1
            print(f"failed: {row['url']}")
        time.sleep(0.2)
        if i % 50 == 0:
            print(f"{i}/{len(rows)}", flush=True)
    print(f"done: {len(rows) - failed}/{len(rows)} files in {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
