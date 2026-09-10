"""Run the static source-code analyzer on the repo and print a report.

    python scripts/code_report.py [--root src] [--json runs/code_report.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu_corruptnet.codeanalysis import analyze_path, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="src")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    reports = analyze_path(args.root)
    s = summarize(reports)

    print(
        f"{s['files']} files · {s['total_loc']} LOC · {s['functions']} functions · "
        f"docstring coverage {s['docstring_coverage'] * 100:.0f}% · "
        f"avg complexity {s['avg_complexity']:.1f}\n"
    )
    print("riskiest functions (cyclomatic complexity):")
    for r in s["riskiest"]:
        print(f"  {r['complexity']:>3}  {r['path']}:{r['line']}  {r['name']}()")
    high = s["high_risk_functions"]
    print(f"\n{len(high)} function(s) at/above the risk threshold (complexity >= 10)")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(s, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
