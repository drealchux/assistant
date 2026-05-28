from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import anthropic

from .config import Settings
from .models import EvalQuestion, EvalResult, EvalSummary
from .prompts import (
    CORRECTNESS_JUDGE_SYSTEM,
    CORRECTNESS_JUDGE_USER,
    FAITHFULNESS_JUDGE_SYSTEM,
    FAITHFULNESS_JUDGE_USER,
)

logger = logging.getLogger(__name__)


# Retrieval Metrics

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    if not relevant_ids:
        return 1.0
    hits = set(retrieved_ids[:k]) & set(relevant_ids)
    return len(hits) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not retrieved_ids[:k]:
        return 0.0
    hits = set(retrieved_ids[:k]) & set(relevant_ids)
    return len(hits) / min(k, len(retrieved_ids))


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain (binary relevance)."""
    relevant_set = set(relevant_ids)

    def dcg(ids: list[str]) -> float:
        return sum(
            1.0 / math.log2(i + 2)
            for i, cid in enumerate(ids[:k])
            if cid in relevant_set
        )

    actual_dcg = dcg(retrieved_ids)
    ideal_dcg = dcg(list(relevant_set)[:k])
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# LLM Judge 

class LLMJudge:
    """Uses Claude as an impartial evaluator for faithfulness and correctness."""

    def __init__(self, settings: Settings):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model

    def score_faithfulness(
        self, question: str, context: str, answer: str
    ) -> tuple[float, str]:
        return self._judge(
            system=FAITHFULNESS_JUDGE_SYSTEM,
            user=FAITHFULNESS_JUDGE_USER.format(
                question=question, context=context, answer=answer
            ),
        )

    def score_correctness(
        self, question: str, gold_answer: str, answer: str
    ) -> tuple[float, str]:
        return self._judge(
            system=CORRECTNESS_JUDGE_SYSTEM,
            user=CORRECTNESS_JUDGE_USER.format(
                question=question, gold_answer=gold_answer, answer=answer
            ),
        )

    def _judge(self, system: str, user: str) -> tuple[float, str]:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            data = json.loads(response.content[0].text)
            return float(data.get("score", 0)), data.get("reasoning", "")
        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            return 0.0, "evaluation error"


# Evaluation Runner 

class EvaluationRunner:
    """Runs the full evaluation pipeline against a benchmark dataset."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.judge = LLMJudge(settings)

    def evaluate_retrieval(
        self,
        question: EvalQuestion,
        retrieved_ids: list[str],
    ) -> dict:
        return {
            "recall_at_5": recall_at_k(retrieved_ids, question.supporting_chunk_ids, 5),
            "recall_at_10": recall_at_k(retrieved_ids, question.supporting_chunk_ids, 10),
            "precision_at_5": precision_at_k(retrieved_ids, question.supporting_chunk_ids, 5),
            "ndcg_at_10": ndcg_at_k(retrieved_ids, question.supporting_chunk_ids, 10),
        }

    def evaluate_answer(
        self,
        question: EvalQuestion,
        predicted_answer: str,
        context: str,
        retrieved_ids: list[str],
        latency_ms: int,
    ) -> EvalResult:
        faithfulness, _ = self.judge.score_faithfulness(
            question.question, context, predicted_answer
        )
        correctness, _ = self.judge.score_correctness(
            question.question, question.gold_answer, predicted_answer
        )
        retrieval_metrics = self.evaluate_retrieval(question, retrieved_ids)

        return EvalResult(
            question_id=question.question_id,
            question=question.question,
            predicted_answer=predicted_answer,
            gold_answer=question.gold_answer,
            retrieved_chunk_ids=retrieved_ids,
            recall_at_5=retrieval_metrics["recall_at_5"],
            recall_at_10=retrieval_metrics["recall_at_10"],
            precision_at_5=retrieval_metrics["precision_at_5"],
            faithfulness_score=faithfulness,
            correctness_score=correctness,
            latency_ms=latency_ms,
        )

    def summarize(self, results: list[EvalResult]) -> EvalSummary:
        n = len(results)
        if n == 0:
            return EvalSummary(
                total_questions=0,
                avg_recall_at_5=0,
                avg_recall_at_10=0,
                avg_precision_at_5=0,
                avg_faithfulness=0,
                avg_correctness=0,
                avg_latency_ms=0,
                hallucination_rate=0,
            )

        return EvalSummary(
            total_questions=n,
            avg_recall_at_5=sum(r.recall_at_5 for r in results) / n,
            avg_recall_at_10=sum(r.recall_at_10 for r in results) / n,
            avg_precision_at_5=sum(r.precision_at_5 for r in results) / n,
            avg_faithfulness=sum(r.faithfulness_score for r in results) / n,
            avg_correctness=sum(r.correctness_score for r in results) / n,
            avg_latency_ms=sum(r.latency_ms for r in results) / n,
            # Hallucination = faithfulness < 3 (below midpoint)
            hallucination_rate=sum(
                1 for r in results if r.faithfulness_score < 3
            ) / n,
        )

    def load_eval_set(self, path: Optional[str] = None) -> list[EvalQuestion]:
        p = Path(path or self.settings.eval_set_path)
        if not p.exists():
            return []
        questions = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(EvalQuestion(**json.loads(line)))
        return questions

    def save_results(self, results: list[EvalResult], output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(r.model_dump_json() + "\n")
