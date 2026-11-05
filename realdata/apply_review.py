"""Apply a second labelling pass to realdata/real_glitches.csv.

    python realdata/apply_review.py decisions1.txt [decisions2.txt ...]

Each decisions file has lines `id | verdict | classes | note` (later files and later lines
win). Verdicts: `ok` (label confirmed), `classes` (still a glitch, classes replaced),
`glitch` (a clean row is really a glitch; classes required), `clean` (a glitch row is
really clean), `drop` (not a usable rendered frame, or undecidable). The CSV keeps every
row for auditability: dropped rows get label `excluded`, and the `review` column records
what the second pass did (confirmed / relabelled / excluded / first-pass-only).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

MANIFEST = Path(__file__).with_name("real_glitches.csv")
CLASSES = {"shader", "shapes", "discoloration", "morse_code", "dotted_lines", "parallel_lines",
           "triangulation", "line_pixelation", "screen_stuttering", "screen_tearing"}


def read_decisions(paths: list[str]) -> dict[str, tuple[str, str, str]]:
    out = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            parts += [""] * (4 - len(parts))
            cid, verdict, classes, note = parts[:4]
            if verdict == "CHECK":
                continue  # resolved by a later line
            bad = [c for c in classes.split(";") if c and c not in CLASSES]
            if bad:
                raise SystemExit(f"{path}: {cid}: unknown classes {bad}")
            out[cid] = (verdict, classes, note)
    return out


def main(paths: list[str]) -> None:
    rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
    decisions = read_decisions(paths)
    stats: dict[str, int] = {}
    for r in rows:
        cid = r["file"].removesuffix(".png")
        r.setdefault("review", "")
        r.setdefault("review_note", "")
        if cid not in decisions:
            r["review"] = r["review"] or "first-pass-only"
            continue
        verdict, classes, note = decisions[cid]
        before = r["label"]
        if verdict == "ok":
            r["review"] = "confirmed"
        elif verdict == "classes":
            if before != "glitch" or not classes:
                raise SystemExit(f"{cid}: 'classes' needs a glitch row and classes")
            r["review"] = "relabelled" if classes != r["classes"] else "confirmed"
            r["classes"] = classes
        elif verdict == "glitch":
            if not classes:
                raise SystemExit(f"{cid}: 'glitch' needs classes")
            r["review"] = "relabelled" if before != "glitch" else "confirmed"
            r["label"], r["classes"] = "glitch", classes
            r["description"] = note or r["description"]
        elif verdict == "clean":
            r["review"] = "relabelled" if before != "clean" else "confirmed"
            r["label"], r["classes"], r["description"] = "clean", "", ""
        elif verdict == "drop":
            r["review"], r["label"] = "excluded", "excluded"
        else:
            raise SystemExit(f"{cid}: unknown verdict '{verdict}'")
        r["review_note"] = note
        key = f"{before}->{r['label']} ({r['review']})"
        stats[key] = stats.get(key, 0) + 1

    fields = ["file", "label", "classes", "description", "review", "review_note",
              "repo", "issue", "url"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)
    for k, v in sorted(stats.items()):
        print(f"{v:4d}  {k}")
    final = {}
    for r in rows:
        final[r["label"]] = final.get(r["label"], 0) + 1
    print("final labels:", final)


if __name__ == "__main__":
    main(sys.argv[1:])
