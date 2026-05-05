"""
Chatbot Evaluation Framework
=============================
A complete auto-evaluation system for the custom chatbot.

Modules:
    scorer        — accuracy, relevance, coherence scoring
    logger        — structured JSON log writer (eval_log.jsonl, bad_responses.jsonl)
    feedback_loop — retraining data exporter (DPO JSONL + human review CSV)
    evaluate      — main CLI runner

Quickstart:
    cd d:/chatbot/evaluation
    python evaluate.py                         # run all 30 test cases
    python evaluate.py --limit 5               # quick smoke test
    python feedback_loop.py                    # export retraining data
"""

from .scorer import evaluate_response, score_accuracy, score_relevance, score_coherence
from .logger import log_evaluation, save_run_summary, load_bad_responses

__all__ = [
    "evaluate_response",
    "score_accuracy",
    "score_relevance",
    "score_coherence",
    "log_evaluation",
    "save_run_summary",
    "load_bad_responses",
]
