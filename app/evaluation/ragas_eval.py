"""
app/evaluation/ragas_eval.py
RAGAS evaluation harness — the core differentiator of this project.

Metrics evaluated:
  - Faithfulness:        Does the answer stick to what the retrieved context says?
  - Answer Relevance:   Does the answer actually address the question asked?
  - Context Precision:  Are the retrieved chunks relevant to the question?
  - Context Recall:     Did we retrieve all the chunks needed to answer? (requires ground truth)

Usage:
  # Run against a dataset file
  python -m app.evaluation.ragas_eval --dataset tests/eval/eval_dataset.json

  # Run programmatically
  from app.evaluation.ragas_eval import run_evaluation
  report = run_evaluation(dataset_path="tests/eval/eval_dataset.json")
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset
from loguru import logger
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.core.config import get_settings
from app.core.rag_chain import query_rag


# ── Eval dataset schema ───────────────────────────────────────────────────────
# Each entry in the JSON file should look like:
# {
#   "question": "What is the refund policy?",
#   "ground_truth": "Customers may return items within 30 days..."  ← optional but enables recall
# }


def load_eval_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} eval samples from {path}")
    return data


def build_ragas_dataset(eval_samples: list[dict]) -> Dataset:
    """
    Run each question through the RAG pipeline and collect inputs/outputs
    in the format RAGAS expects.
    """
    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, sample in enumerate(eval_samples):
        question = sample["question"]
        ground_truth = sample.get("ground_truth", "")

        logger.info(f"  [{i+1}/{len(eval_samples)}] Evaluating: {question[:60]}...")

        response = query_rag(question)

        rows["question"].append(question)
        rows["answer"].append(response.answer)
        rows["contexts"].append([doc.page_content for doc in response.retrieved_chunks])
        rows["ground_truth"].append(ground_truth)

    return Dataset.from_dict(rows)


def run_evaluation(
    dataset_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run RAGAS evaluation and return a results dict with pass/fail gates.

    Args:
        dataset_path: Path to JSON eval dataset. Defaults to settings value.
        output_dir: Where to write the report CSV. Defaults to ./eval_results/.

    Returns:
        dict with scores, pass/fail status, and output file path.
    """
    settings = get_settings()
    dataset_path = dataset_path or settings.ragas_eval_dataset_path
    output_dir = output_dir or Path("./eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting RAGAS evaluation run")
    logger.info(f"Dataset: {dataset_path}")
    logger.info("=" * 60)

    eval_samples = load_eval_dataset(dataset_path)
    ragas_dataset = build_ragas_dataset(eval_samples)

    # ── Choose metrics based on whether ground truth is available ─
    has_ground_truth = any(s.get("ground_truth") for s in eval_samples)
    metrics = [faithfulness, answer_relevancy, context_precision]
    if has_ground_truth:
        metrics.append(context_recall)
        logger.info("Ground truth detected — context recall will be evaluated")
    else:
        logger.info("No ground truth — skipping context recall")

    logger.info(f"Running RAGAS with metrics: {[m.name for m in metrics]}")
    results = evaluate(ragas_dataset, metrics=metrics)

    scores = results.to_pandas()
    mean_scores = scores.mean(numeric_only=True).to_dict()

    # ── CI gate evaluation ────────────────────────────────────────
    faithfulness_score = mean_scores.get("faithfulness", 0.0)
    hallucination_rate = 1.0 - faithfulness_score  # inverse of faithfulness
    answer_relevancy_score = mean_scores.get("answer_relevancy", 0.0)
    context_precision_score = mean_scores.get("context_precision", 0.0)

    gates_passed = (
        hallucination_rate <= settings.hallucination_threshold
        and faithfulness_score >= settings.faithfulness_threshold
    )

    # ── Write report ──────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = output_dir / f"eval_report_{timestamp}.csv"
    scores.to_csv(report_path, index=False)

    summary_path = output_dir / "latest_summary.json"
    summary = {
        "timestamp": timestamp,
        "dataset_size": len(eval_samples),
        "scores": {
            "faithfulness": round(faithfulness_score, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "answer_relevancy": round(answer_relevancy_score, 4),
            "context_precision": round(context_precision_score, 4),
        },
        "thresholds": {
            "hallucination_threshold": settings.hallucination_threshold,
            "faithfulness_threshold": settings.faithfulness_threshold,
        },
        "gates_passed": gates_passed,
        "report_path": str(report_path),
        "chunk_config": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.retriever_top_k,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    # ── Console report ────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Faithfulness:      {faithfulness_score:.1%}")
    logger.info(f"  Hallucination rate:{hallucination_rate:.1%}  (target: ≤{settings.hallucination_threshold:.0%})")
    logger.info(f"  Answer Relevancy:  {answer_relevancy_score:.1%}")
    logger.info(f"  Context Precision: {context_precision_score:.1%}")
    logger.info("-" * 60)
    logger.info(f"  CI Gates: {'✓ PASSED' if gates_passed else '✗ FAILED'}")
    logger.info("=" * 60)

    if not gates_passed:
        logger.error(
            f"Evaluation gates FAILED. "
            f"Hallucination={hallucination_rate:.1%} (max {settings.hallucination_threshold:.0%}), "
            f"Faithfulness={faithfulness_score:.1%} (min {settings.faithfulness_threshold:.0%})"
        )

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--dataset", type=Path, help="Path to eval dataset JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("./eval_results"))
    args = parser.parse_args()

    result = run_evaluation(dataset_path=args.dataset, output_dir=args.output_dir)

    import sys
    sys.exit(0 if result["gates_passed"] else 1)
