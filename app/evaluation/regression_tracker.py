"""
app/evaluation/regression_tracker.py
Tracks evaluation metrics across runs to detect regressions and show improvements.

This is what generates the "hallucination rate dropped from 23% to 4%" chart
that makes the README compelling. Each eval run appends to a JSONL history file.
The tracker compares the latest run against baseline and flags regressions.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


HISTORY_FILE = Path("./eval_results/metrics_history.jsonl")
BASELINE_FILE = Path("./eval_results/baseline.json")


def record_run(summary: dict) -> None:
    """Append an eval summary to the metrics history file."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(summary) + "\n")
    logger.info(f"Recorded eval run to {HISTORY_FILE}")


def set_baseline(summary: dict) -> None:
    """Mark the current run as the baseline for regression comparison."""
    BASELINE_FILE.write_text(json.dumps(summary, indent=2))
    logger.info(f"Baseline set: faithfulness={summary['scores']['faithfulness']:.1%}")


def load_history() -> list[dict]:
    """Load all recorded eval runs."""
    if not HISTORY_FILE.exists():
        return []
    runs = []
    with open(HISTORY_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def load_baseline() -> Optional[dict]:
    if not BASELINE_FILE.exists():
        return None
    return json.loads(BASELINE_FILE.read_text())


def check_regression(current: dict, tolerance: float = 0.05) -> dict:
    """
    Compare current run to baseline.
    Returns regression report with pass/fail per metric.

    Args:
        current: Current eval summary dict.
        tolerance: Allowed score drop before flagging as regression (default 5%).
    """
    baseline = load_baseline()
    if not baseline:
        logger.warning("No baseline set — skipping regression check")
        return {"status": "no_baseline", "regressions": []}

    regressions = []
    improvements = []

    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        base_score = baseline["scores"].get(metric, 0)
        curr_score = current["scores"].get(metric, 0)
        delta = curr_score - base_score

        if delta < -tolerance:
            regressions.append({
                "metric": metric,
                "baseline": base_score,
                "current": curr_score,
                "delta": delta,
            })
            logger.error(f"REGRESSION: {metric} dropped {abs(delta):.1%} ({base_score:.1%} → {curr_score:.1%})")
        elif delta > tolerance:
            improvements.append({
                "metric": metric,
                "baseline": base_score,
                "current": curr_score,
                "delta": delta,
            })
            logger.info(f"IMPROVEMENT: {metric} improved {delta:.1%} ({base_score:.1%} → {curr_score:.1%})")

    return {
        "status": "regression_found" if regressions else "ok",
        "regressions": regressions,
        "improvements": improvements,
        "baseline_timestamp": baseline.get("timestamp"),
        "current_timestamp": current.get("timestamp"),
    }


def generate_trend_report(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Build a DataFrame of all eval runs suitable for charting.
    Columns: timestamp, faithfulness, hallucination_rate, answer_relevancy,
             context_precision, chunk_size, chunk_overlap, top_k
    """
    history = load_history()
    if not history:
        logger.warning("No eval history found — run evaluations first")
        return pd.DataFrame()

    rows = []
    for run in history:
        row = {
            "timestamp": run.get("timestamp", ""),
            "dataset_size": run.get("dataset_size", 0),
            **run.get("scores", {}),
            **{f"config_{k}": v for k, v in run.get("chunk_config", {}).items()},
            "gates_passed": run.get("gates_passed", False),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%dT%H%M%S", utc=True)
    df = df.sort_values("timestamp")

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Trend report written to {output_path}")

    return df


def print_trend_summary() -> None:
    """Print a human-readable trend summary to console."""
    df = generate_trend_report()
    if df.empty:
        print("No eval history found.")
        return

    print("\n" + "=" * 70)
    print("EVALUATION TREND SUMMARY")
    print("=" * 70)
    print(f"Total runs: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print()

    for metric in ["faithfulness", "hallucination_rate", "answer_relevancy", "context_precision"]:
        if metric not in df.columns:
            continue
        first = df[metric].iloc[0]
        latest = df[metric].iloc[-1]
        delta = latest - first
        direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"  {metric:<22} {first:.1%} → {latest:.1%}  {direction} {abs(delta):.1%}")

    print("=" * 70)


if __name__ == "__main__":
    print_trend_summary()
