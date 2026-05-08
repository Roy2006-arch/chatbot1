#!/usr/bin/env python3
"""
Dataset Quality Evaluation Script

Analyzes a dataset for quality metrics, distributions, and issues.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.ingestion import DataIngestor, DatasetExample
from src.quality.scoring import QualityScorer
from src.quality.deduplication import Deduplicator
from src.pipeline.validation import CodeValidator, MarkdownValidator


def analyze_dataset(examples, scorer, code_val, md_val):
    report = {
        "total": len(examples),
        "avg_length": {
            "instruction": sum(len(e.instruction) for e in examples) / max(len(examples), 1),
            "output": sum(len(e.output) for e in examples) / max(len(examples), 1),
        },
        "category_distribution": dict(Counter(e.category for e in examples)),
        "difficulty_distribution": dict(Counter(e.difficulty for e in examples)),
        "quality_stats": {},
        "code_quality": {},
        "markdown_quality": {},
    }

    if examples:
        scores = [e.quality_score for e in examples]
        report["quality_stats"] = {
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "below_threshold": sum(1 for s in scores if s < 0.65),
        }

    code_blocks = sum(1 for e in examples if "```" in e.output)
    report["code_quality"] = {
        "with_code_blocks": code_blocks,
        "percentage": round(code_blocks / max(len(examples), 1) * 100, 2),
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate dataset quality")
    parser.add_argument("input", help="Input dataset path (file or directory)")
    parser.add_argument("--report", default="quality_report.json", help="Output report path")
    args = parser.parse_args()

    ingestor = DataIngestor()
    scorer = QualityScorer()
    code_val = CodeValidator()
    md_val = MarkdownValidator()

    print(f"Ingesting data from: {args.input}")
    examples = list(ingestor.ingest_from_path(args.input))

    if not examples:
        print("No examples found!")
        return

    print(f"Analyzing {len(examples)} examples...")

    examples = scorer.score(examples)
    report = analyze_dataset(examples, scorer, code_val, md_val)

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nQuality Report saved to: {args.report}")
    print(f"Total: {report['total']}")
    print(f"Categories: {json.dumps(report['category_distribution'], indent=2)}")
    print(f"Quality - Mean: {report['quality_stats'].get('mean', 'N/A'):.3f}, "
          f"Below threshold: {report['quality_stats'].get('below_threshold', 'N/A')}")


if __name__ == "__main__":
    main()
