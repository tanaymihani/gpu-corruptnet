"""Collect candidate real-glitch screenshots from public bug reports (GitHub issues of
graphics translation layers, emulators and open-source engines).

    python realdata/find_glitch_reports.py --out data/glitch_candidates.csv

Needs an authenticated `gh` CLI (search API). Writes one row per (issue, image). The
images stay on GitHub: realdata/real_glitches.csv keeps only URLs + labels, so nothing
third-party is redistributed in this repo. Labels are added by hand after review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time

QUERIES = [
    ("doitsujin/dxvk", "rendering"), ("doitsujin/dxvk", "artifacts"),
    ("doitsujin/dxvk", "graphical glitches"), ("doitsujin/dxvk", "corruption"),
    ("HansKristian-Work/vkd3d-proton", "rendering"),
    ("HansKristian-Work/vkd3d-proton", "artifacts"),
    ("HansKristian-Work/vkd3d-proton", "graphical glitches"),
    ("ValveSoftware/Proton", "graphical glitches"), ("ValveSoftware/Proton", "artifacts"),
    ("godotengine/godot", "rendering artifacts"), ("godotengine/godot", "graphical glitch"),
    ("KhronosGroup/MoltenVK", "artifacts"), ("KhronosGroup/MoltenVK", "rendering"),
    ("RPCS3/rpcs3", "graphical"), ("PCSX2/pcsx2", "graphical"),
    ("hrydgard/ppsspp", "graphical glitch"), ("xemu-project/xemu", "rendering"),
    ("supertuxkart/stk-code", "graphics"), ("luanti-org/luanti", "rendering"),
    ("flightgear/flightgear", "rendering"), ("Ryujinx/Ryujinx", "graphical"),
    ("citra-emu/citra", "graphical"), ("dolphin-emu/dolphin", "graphical"),
    ("bevyengine/bevy", "artifacts"), ("gfx-rs/wgpu", "corruption"),
]
IMG = re.compile(
    r"(https://(?:user-images\.githubusercontent\.com|github\.com/user-attachments/assets|"
    r"private-user-images\.githubusercontent\.com)/[^\s\)\"'>]+)"
)


def search(repo: str, words: str, pages: int = 1) -> list[dict]:
    items = []
    for page in range(1, pages + 1):
        q = f"repo:{repo} is:issue {words} in:title,body"
        out = subprocess.run(
            ["gh", "api", "-X", "GET", "search/issues", "-f", f"q={q}", "-f", "per_page=100",
             "-f", f"page={page}"],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            print(f"search failed for {q}: {out.stderr[:200]}")
            break
        items += json.loads(out.stdout).get("items", [])
        time.sleep(2.5)  # search API: 30 requests / minute
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/glitch_candidates.csv")
    ap.add_argument("--per-issue", type=int, default=2)
    args = ap.parse_args()
    rows, seen = [], set()
    for repo, words in QUERIES:
        items = search(repo, words)
        n0 = len(rows)
        for it in items:
            for url in IMG.findall(it.get("body") or "")[: args.per_issue]:
                url = url.rstrip(".,")
                if url in seen:
                    continue
                seen.add(url)
                rows.append({"repo": repo, "issue": it["html_url"], "issue_title": it["title"],
                             "image_url": url})
        print(f"{repo:35s} {words:20s} issues={len(items):3d} images+={len(rows) - n0}",
              flush=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["repo", "issue", "issue_title", "image_url"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} candidates to {args.out}")


if __name__ == "__main__":
    main()
