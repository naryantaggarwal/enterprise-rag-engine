#!/usr/bin/env python3
"""
scripts/ingest_eval_docs.py
Ingest sample documents before running RAGAS evaluation in CI.
The eval dataset questions must have corresponding source content in the vector store.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.ingestion import ingest_directory

EVAL_DOCS_DIR = Path(os.getenv("DOCS_DIR", "tests/eval/sample_docs"))

def main():
    if not EVAL_DOCS_DIR.exists():
        print(f"Eval docs directory not found: {EVAL_DOCS_DIR}")
        print("Creating it — add your sample PDFs/TXTs there for eval context")
        EVAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        sys.exit(0)

    results = ingest_directory(EVAL_DOCS_DIR)
    ingested = sum(1 for r in results if r["status"] == "ingested")
    print(f"Ingested {ingested} document(s) for evaluation")

if __name__ == "__main__":
    main()
