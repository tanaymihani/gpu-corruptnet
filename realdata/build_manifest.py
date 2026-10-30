"""Build realdata/clean_frames.csv: free-licence screenshots of open-source games from
Wikimedia Commons, with author + licence per file (so the frames are redistributable).

    python realdata/build_manifest.py --per-title 40

Only files >= 1024 px wide under CC0 / CC BY / CC BY-SA / GPL / public-domain terms are
kept. Download URLs are capped at 1920 px wide (Commons serves a scaled copy above that).
Whole titles marked "unseen" are held out of training (unseen content).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "gpu-corruptnet-research/0.1 (https://github.com/tanaymihani/gpu-corruptnet)"

# title -> (Commons categories, split). 3D titles dominate on purpose: that is where GPU
# rendering faults show up. Four held-out titles span voxel 3D, racing, ball-physics and 2D.
TITLES: dict[str, tuple[list[str], str]] = {
    "SuperTuxKart": (["Screenshots of SuperTuxKart"], "seen"),
    "0 A.D.": (["Screenshots of 0 A.D."], "seen"),
    "Xonotic": (["Screenshots of Xonotic", "Xonotic screenshots"], "seen"),
    "Luanti (Minetest)": (["Screenshots of Luanti", "Screenshots of Minetest"], "seen"),
    "Warzone 2100": (["Screenshots of Warzone 2100"], "seen"),
    "Red Eclipse": (["Red Eclipse screenshots", "Red Eclipse (video game)"], "seen"),
    "OpenArena": (["Screenshots of OpenArena"], "seen"),
    "FlightGear": (["FlightGear"], "seen"),
    "Cube 2: Sauerbraten": (["Cube 2: Sauerbraten", "Sauerbraten (video game)"], "seen"),
    "TORCS": (["TORCS"], "seen"),
    "Vega Strike": (["Screenshots of Vega Strike"], "seen"),
    "AssaultCube": (["Screenshots of AssaultCube"], "seen"),
    "Nexuiz": (["Screenshots of Nexuiz"], "seen"),
    "Tremulous / Unvanquished": (["Tremulous", "Unvanquished"], "seen"),
    "Yo Frankie!": (["Screenshots of Yo Frankie!"], "seen"),
    "Pioneer": (["Pioneer (video game)"], "seen"),
    "Neverball": (["Screenshots of Neverball", "Neverball"], "unseen"),
    "Veloren": (["Screenshots of Veloren", "Veloren"], "unseen"),
    "Speed Dreams": (["Screenshots of Speed Dreams"], "unseen"),
    "Battle for Wesnoth": (["Screenshots of Battle for Wesnoth"], "unseen"),
}
FREE = re.compile(r"^(CC0|CC BY(-SA)?( \d\.\d)?|GPL|LGPL|Public domain|PD|FAL|MIT|BSD)", re.I)


def api(**params) -> dict:
    params |= {"format": "json", "formatversion": "2"}
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001 - transient network errors: back off and retry
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Commons API failed: {params}")


def category_files(cat: str, depth: int = 1) -> list[str]:
    files, subcats, cont = [], [], {}
    while True:
        d = api(action="query", list="categorymembers", cmtitle=f"Category:{cat}",
                cmtype="file|subcat", cmlimit=500, **cont)
        for m in d["query"]["categorymembers"]:
            (files if m["ns"] == 6 else subcats).append(m["title"])
        if "continue" not in d:
            break
        cont = d["continue"]
    if depth > 0:
        for sc in subcats:
            files += category_files(sc.removeprefix("Category:"), depth - 1)
    return files


def image_info(titles: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(titles), 40):
        d = api(action="query", prop="imageinfo", titles="|".join(titles[i : i + 40]),
                iiprop="url|size|mime|extmetadata", iiurlwidth=1920)
        for p in d["query"]["pages"]:
            if "imageinfo" in p:
                out.append({"title": p["title"], **p["imageinfo"][0]})
        time.sleep(0.3)
    return out


def clean_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-title", type=int, default=40)
    ap.add_argument("--out", default="realdata/clean_frames.csv")
    args = ap.parse_args()

    rows = []
    for title, (cats, split) in TITLES.items():
        names = sorted({f for c in cats for f in category_files(c)})
        infos = image_info(names)
        kept = []
        for info in infos:
            meta = info.get("extmetadata", {})
            lic = clean_html(meta.get("LicenseShortName", {}).get("value", ""))
            if info["mime"] not in ("image/png", "image/jpeg") or info["width"] < 1024:
                continue
            if info["height"] < 540 or not FREE.match(lic):
                continue
            url = info.get("thumburl") if info["width"] > 1920 else info["url"]
            kept.append({
                "title": title,
                "split": split,
                "commons_file": info["title"].removeprefix("File:"),
                "width": info.get("thumbwidth") if info["width"] > 1920 else info["width"],
                "height": info.get("thumbheight") if info["width"] > 1920 else info["height"],
                "license": lic,
                "artist": clean_html(meta.get("Artist", {}).get("value", ""))[:120],
                "source_page": info["descriptionurl"],
                "url": url,
            })
        kept = kept[: args.per_title]  # names were sorted: deterministic
        print(f"{title:20s} {len(names):4d} files -> {len(kept):3d} kept", flush=True)
        rows += kept

    for i, r in enumerate(rows):
        ext = ".png" if r["url"].lower().endswith(".png") else ".jpg"
        r["file"] = f"{i:04d}{ext}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "title", "split", "width", "height", "license", "artist",
              "commons_file", "source_page", "url"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
