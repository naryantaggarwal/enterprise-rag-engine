#!/usr/bin/env python3
"""
scripts/check_regression.py
CI script: compare latest eval run against baseline.
Exits 1 if any metric regressed beyond tolerance.
Run after ragas_eval.py in CI.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.regression_tracker import check_regression, load_history

def main():
    history = load_history()
    if not history:
        print("No eval history — skipping regression check")
        sys.exit(0)

    current = history[-1]
    report = check_regression(current, tolerance=0.05)

    if report["status"] == "no_baseline":
        print("No baseline set — this run will become the baseline")
        from app.evaluation.regression_tracker import set_baseline
        set_baseline(current)
        sys.exit(0)

    if report["regressions"]:
        print("\n❌ REGRESSIONS DETECTED:")
        for r in report["regressions"]:
            print(f"  {r['metric']}: {r['baseline']:.1%} → {r['current']:.1%} (Δ {r['delta']:+.1%})")
        sys.exit(1)
    else:
        print("✓ No regressions detected")
        if report["improvements"]:
            print("\n🚀 Improvements:")
            for i in report["improvements"]:
                print(f"  {i['metric']}: {i['baseline']:.1%} → {i['current']:.1%} (Δ {i['delta']:+.1%})")
        sys.exit(0)

if __name__ == "__main__":
    main()
