"""
scorer.py
---------
Core scoring engine for the chatbot evaluation framework.
Computes three primary dimensions:
  - Accuracy   : keyword/fact coverage vs. expected answer
  - Relevance  : semantic similarity between prompt and response
  - Coherence  : structural quality signals (length, grammar cues, repetition)
"""

import re
import math
from typing import Optional
from sentence_transformers import SentenceTransformer, util

# Shared model instance (loaded once at import time)
_EMBEDDER: Optional[SentenceTransformer] = None


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


# ---------------------------------------------------------------------------
# Individual Scoring Functions
# ---------------------------------------------------------------------------

def score_accuracy(response: str, expected_keywords: list[str]) -> float:
    """
    Measures factual coverage.
    Returns 0.0–1.0: fraction of expected keywords found in the response (case-insensitive).
    If no expected_keywords are supplied, returns None (unscored).
    """
    if not expected_keywords:
        return None
    response_lower = response.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    return round(hits / len(expected_keywords), 4)


def score_relevance(prompt: str, response: str) -> float:
    """
    Measures semantic similarity between the user prompt and the bot response
    using cosine similarity on sentence embeddings.
    Returns 0.0–1.0.
    """
    if not response.strip():
        return 0.0
    embedder = _get_embedder()
    emb_prompt = embedder.encode(prompt, convert_to_tensor=True)
    emb_response = embedder.encode(response, convert_to_tensor=True)
    similarity = util.cos_sim(emb_prompt, emb_response).item()
    # Clamp to [0, 1]
    return round(max(0.0, min(1.0, similarity)), 4)


def score_coherence(response: str) -> float:
    """
    Heuristic coherence score based on structural quality signals.
    Penalises: empty responses, excessive repetition, very short responses,
               runaway length (hallucination proxy), broken sentence structure.
    Returns 0.0–1.0.
    """
    if not response.strip():
        return 0.0

    score = 1.0
    words = response.split()
    num_words = len(words)

    # --- Penalty: too short (< 5 words) ---
    if num_words < 5:
        score -= 0.4

    # --- Penalty: suspiciously long (> 500 words — likely hallucinating) ---
    if num_words > 500:
        score -= 0.2

    # --- Penalty: high repetition ratio ---
    unique_ratio = len(set(words)) / max(num_words, 1)
    if unique_ratio < 0.35:
        score -= 0.25
    elif unique_ratio < 0.50:
        score -= 0.10

    # --- Penalty: contains hallucination-style role leakage ---
    if re.search(r'\b(user:|assistant:|system:)\b', response, re.IGNORECASE):
        score -= 0.30

    # --- Penalty: no sentence-ending punctuation at all ---
    if not re.search(r'[.!?]', response):
        score -= 0.10

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Composite Grade
# ---------------------------------------------------------------------------

WEIGHTS = {
    "accuracy":  0.40,   # factual correctness is most important
    "relevance": 0.35,   # staying on-topic
    "coherence": 0.25,   # readable / well-formed output
}

GRADE_THRESHOLDS = [
    (0.85, "A", "Excellent"),
    (0.70, "B", "Good"),
    (0.55, "C", "Acceptable"),
    (0.40, "D", "Poor"),
    (0.00, "F", "Failing"),
]


def compute_composite(accuracy: Optional[float], relevance: float, coherence: float) -> dict:
    """
    Weighted composite score.
    If accuracy is None (no expected keywords), re-weights between relevance & coherence.
    """
    if accuracy is None:
        rel_w = WEIGHTS["relevance"] / (WEIGHTS["relevance"] + WEIGHTS["coherence"])
        coh_w = 1.0 - rel_w
        composite = round(relevance * rel_w + coherence * coh_w, 4)
    else:
        composite = round(
            accuracy  * WEIGHTS["accuracy"] +
            relevance * WEIGHTS["relevance"] +
            coherence * WEIGHTS["coherence"],
            4,
        )

    grade, label = "F", "Failing"
    for threshold, g, l in GRADE_THRESHOLDS:
        if composite >= threshold:
            grade, label = g, l
            break

    return {
        "composite_score": composite,
        "grade": grade,
        "grade_label": label,
    }


# ---------------------------------------------------------------------------
# Main Public API
# ---------------------------------------------------------------------------

def evaluate_response(
    prompt: str,
    response: str,
    expected_keywords: Optional[list[str]] = None,
) -> dict:
    """
    Full evaluation of a single chatbot response.

    Args:
        prompt            : The user's input message.
        response          : The chatbot's output.
        expected_keywords : Optional list of keywords/facts that should appear.

    Returns:
        A flat dict containing all scores and the composite grade.
    """
    accuracy  = score_accuracy(response, expected_keywords or [])
    relevance = score_relevance(prompt, response)
    coherence = score_coherence(response)
    composite = compute_composite(accuracy, relevance, coherence)

    return {
        "scores": {
            "accuracy":  accuracy,
            "relevance": relevance,
            "coherence": coherence,
        },
        **composite,
    }
