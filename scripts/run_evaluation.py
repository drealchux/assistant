#!/usr/bin/env python3
"""
Run the evaluation pipeline against the benchmark eval set.

Usage:
    python -m scripts.run_evaluation
    python -m scripts.run_evaluation --eval-set data/eval_set.jsonl --output data/eval_results.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import get_settings
from backend.evaluation import EvaluationRunner
from backend.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run_eval(args: argparse.Namespace) -> None:
    settings = get_settings()
    pipeline = RAGPipeline(settings)
    runner = EvaluationRunner(settings)

    eval_questions = runner.load_eval_set(args.eval_set)
    if not eval_questions:
        print(f"No evaluation questions found in {args.eval_set}")
        return

    print(f"Running evaluation on {len(eval_questions)} questions...")
    results = []

    for i, question in enumerate(eval_questions, 1):
        t0 = time.time()
        try:
            response = await pipeline.query(
                query=question.question,
                user_id="eval_runner",
                user_groups=["__eval__"],
                top_k=10,
            )
            retrieved_ids = [c.chunk_id for c in response.citations]
            context = "\n\n".join(c.relevant_text for c in response.citations)
            latency = int((time.time() - t0) * 1000)

            result = runner.evaluate_answer(
                question=question,
                predicted_answer=response.answer,
                context=context,
                retrieved_ids=retrieved_ids,
                latency_ms=latency,
            )
            results.append(result)
            print(
                f"[{i}/{len(eval_questions)}] {question.question[:60]}... "
                f"R@5={result.recall_at_5:.2f} Faith={result.faithfulness_score:.1f}"
            )
        except Exception as e:
            logger.error(f"Error on question {question.question_id}: {e}")

    summary = runner.summarize(results)
    runner.save_results(results, args.output)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions:      {summary.total_questions}")
    print(f"Avg Recall@5:         {summary.avg_recall_at_5:.3f}  (target ≥ 0.90)")
    print(f"Avg Recall@10:        {summary.avg_recall_at_10:.3f}  (target ≥ 0.80)")
    print(f"Avg Precision@5:      {summary.avg_precision_at_5:.3f}")
    print(f"Avg Faithfulness:     {summary.avg_faithfulness:.2f}/5 (target ≥ 4.0)")
    print(f"Avg Correctness:      {summary.avg_correctness:.2f}/5 (target ≥ 4.0)")
    print(f"Avg Latency:          {summary.avg_latency_ms:.0f}ms")
    print(f"Hallucination rate:   {summary.hallucination_rate:.1%}  (target < 5%)")
    print("=" * 60)

    # Flag targets not met
    checks = [
        ("Recall@5 ≥ 0.90", summary.avg_recall_at_5 >= 0.90),
        ("Faithfulness ≥ 4.0", summary.avg_faithfulness >= 4.0),
        ("Correctness ≥ 4.0", summary.avg_correctness >= 4.0),
        ("Hallucination < 5%", summary.hallucination_rate < 0.05),
    ]
    for label, passed in checks:
        print(f"  {'✓' if passed else '✗'} {label}")

    summary_path = Path(args.output).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary.model_dump(), f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--eval-set",
        default="data/eval_set.jsonl",
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--output",
        default="data/eval_results.jsonl",
        help="Output path for per-question results",
    )
    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
