"""
feedback/connector.py
---------------------
Bridges self-improvement pipeline output into MistakeMemory corrections
and curriculum dataset builder for downstream training.

Data flow:
  1. Run SelfImprovementPipeline (or load existing ImprovementReport)
  2. Push high-quality corrections into MistakeMemory with SELF_IMPROVEMENT source
  3. Export DPO-format pairs to a known path for retraining_pipeline.py consumption
  4. Build curriculum-stage-tagged dataset for DatasetBridge
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any

from .db_schema import get_conn, _now_utc
from .mistake_memory import MistakeMemory
from self_improvement.schema import (
    SelfImprovementExample, CorrectionMethod, ExampleSource, ImprovementReport
)
from self_improvement.pipeline import SelfImprovementPipeline

log = logging.getLogger("chatbot.feedback_connector")

_IMPROVEMENT_SOURCE = "self_improvement"


def push_corrections_to_mistake_memory(
    examples: List[SelfImprovementExample],
    min_quality: float = 0.7,
) -> int:
    """
    Push high-quality corrections from self-improvement into MistakeMemory
    so they appear as anti-mistake context for future queries.

    Only examples with quality_score >= min_quality are injected.
    """
    mm = MistakeMemory()
    pushed = 0

    for ex in examples:
        if ex.quality_score < min_quality:
            continue
        if not ex.corrected_response or len(ex.corrected_response.strip()) < 10:
            continue

        mm.record_failure(
            conv_id=ex.metadata.get("failed_query_id", f"si_{ex.id}"),
            session_id="self_improvement",
            prompt=ex.prompt,
            response=ex.original_response or "",
            source=_IMPROVEMENT_SOURCE,
            composite_score=ex.quality_score,
            failure_reasons=ex.failure_reasons + ["Corrected by self-improvement pipeline"],
        )

        conn = get_conn()
        conn.execute(
            """
            UPDATE failed_queries
            SET preferred_response = ?, resolved = 1
            WHERE prompt = ? AND source = ?
            """,
            (ex.corrected_response, ex.prompt[:200], _IMPROVEMENT_SOURCE),
        )
        conn.commit()

        pushed += 1

    log.info("[Connector] Pushed %d corrections into MistakeMemory", pushed)
    return pushed


def export_curriculum_dataset(
    examples: List[SelfImprovementExample],
    output_dir: str,
) -> Dict[str, str]:
    """
    Build curriculum-stage-tagged dataset from self-improvement examples.
    Stages map difficulty -> stage:
      difficulty 1 -> stage 1 (basic)
      difficulty 2 -> stage 2 (intermediate)
      difficulty 3 -> stage 3 (advanced)
      difficulty 4 -> stage 4 (expert)
    """
    from self_improvement.dataset_builder import DatasetBuilder

    os.makedirs(output_dir, exist_ok=True)

    builder = DatasetBuilder({"curriculum_order": True, "include_metadata": True})
    dataset = builder.build_dataset(
        correction_examples=examples,
        quality_examples=[],
        hard_examples=[],
        max_total=len(examples),
    )
    train, val, test = builder.split_dataset(dataset)

    paths = {}
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        stage_groups = {1: [], 2: [], 3: [], 4: []}
        for ex in split_data:
            stage = min(max(ex.difficulty, 1), 4)
            stage_groups[stage].append(ex)

        for stage, group in stage_groups.items():
            if not group:
                continue
            path = builder.export_transformers(
                group,
                os.path.join(output_dir, f"{split_name}_stage{stage}.jsonl"),
            )
            paths[f"{split_name}_stage{stage}"] = path

    paths["train_combined"] = builder.export_transformers(
        train, os.path.join(output_dir, "train_combined.jsonl")
    )
    paths["dpo_pairs"] = builder.export_dpo(
        [ex for ex in examples if ex.original_response and ex.corrected_response],
        os.path.join(output_dir, "dpo_pairs.jsonl"),
    )

    log.info("[Connector] Curriculum dataset exported to %s (%d files)", output_dir, len(paths))
    return paths


def run_and_connect(
    max_failed: int = 500,
    max_corrections: int = 200,
    output_dir: Optional[str] = None,
    min_quality: float = 0.7,
) -> Dict[str, Any]:
    """
    Run the full self-improvement pipeline and connect outputs into
    MistakeMemory and curriculum dataset exports.

    Returns a summary dict with run statistics.
    """
    pipeline = SelfImprovementPipeline()
    report = pipeline.run(
        max_failed=max_failed,
        max_corrections=max_corrections,
        output_dir=output_dir,
    )

    corrections = pipeline._run_corrections(max_failed, max_corrections)
    quality = pipeline._run_quality_curation(max_corrections)
    hard = pipeline._run_hard_example_mining(max_corrections)
    all_examples = corrections + quality + hard

    summary = {
        "run_id": report.run_id,
        "corrections_generated": report.corrections_generated,
        "mistake_memory_pushed": 0,
        "curriculum_exports": {},
    }

    pushed = push_corrections_to_mistake_memory(all_examples, min_quality)
    summary["mistake_memory_pushed"] = pushed

    cur_output = output_dir or "self_improvement_data/curriculum"
    cur_paths = export_curriculum_dataset(all_examples, cur_output)
    summary["curriculum_exports"] = cur_paths

    return summary


def get_curriculum_data_paths(base_dir: str = "self_improvement_data/curriculum") -> Dict[str, str]:
    """Return paths to curriculum dataset files for training scripts."""
    import glob
    paths = {}
    for pattern in ["train_combined.jsonl", "dpo_pairs.jsonl",
                     "train_stage*.jsonl", "val_stage*.jsonl", "test_stage*.jsonl"]:
        full_pattern = os.path.join(base_dir, pattern)
        matches = sorted(glob.glob(full_pattern))
        for m in matches:
            key = os.path.splitext(os.path.basename(m))[0]
            paths[key] = m
    return paths
