#!/usr/bin/env python3
"""
Advanced Chatbot Dataset Pipeline Runner

Usage:
    python run_pipeline.py --config config/pipeline.yaml --output exports/my_dataset
    python run_pipeline.py --categories competitive_programming,debugging --huggingface
    python run_pipeline.py --synthetic-only --count 50000
"""

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.orchestrator import DatasetPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Chatbot Dataset Pipeline")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Pipeline config path")
    parser.add_argument("--output", default="exports/dataset", help="Output directory")
    parser.add_argument("--categories", default=None, help="Comma-separated categories to process")
    parser.add_argument("--huggingface", action="store_true", help="Load datasets from HuggingFace")
    parser.add_argument("--framework", default="transformers", choices=["transformers", "axolotl", "openai"], help="Export framework format")
    parser.add_argument("--format", default="jsonl", choices=["jsonl", "json"], help="Export file format")
    parser.add_argument("--synthetic-only", action="store_true", help="Generate synthetic data only")
    parser.add_argument("--count", type=int, default=10000, help="Number of synthetic examples")
    parser.add_argument("--no-synthetic", action="store_true", help="Skip synthetic data generation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--report", default="pipeline_report.json", help="Output report path")
    return parser.parse_args()


def run_synthetic_only(pipeline, args):
    from src.pipeline.ingestion import DatasetExample

    categories = args.categories.split(",") if args.categories else list(pipeline.config.get("categories", {}).keys())
    per_category = args.count // max(len(categories), 1)
    all_examples = []

    for cat in categories:
        cat = cat.strip()
        print(f"Generating {per_category} synthetic examples for: {cat}")
        examples = pipeline.synthetic_generator.generate(cat, count=per_category)
        all_examples.extend(examples)

    print(f"Total synthetic examples: {len(all_examples)}")

    pipeline._stage_quality_scoring(all_examples)
    pipeline._stage_deduplication(all_examples)
    pipeline._stage_filtering(all_examples)
    pipeline._stage_curriculum_scoring(all_examples)

    pipeline.exporter.export_for_training(all_examples, args.output, framework=args.framework)
    pipeline.exporter.export_metadata(all_examples, args.output)

    return all_examples


def main():
    args = parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    pipeline = DatasetPipeline(config_path=args.config)
    pipeline.logger.info("=" * 60)
    pipeline.logger.info("Starting Advanced Chatbot Dataset Pipeline")
    pipeline.logger.info("=" * 60)

    if args.synthetic_only:
        examples = run_synthetic_only(pipeline, args)
    else:
        pipeline_kwargs = {
            "output_dir": args.output,
            "format": args.format,
            "framework": args.framework,
            "load_huggingface": args.huggingface,
        }

        if args.categories:
            pipeline_kwargs["categories"] = [c.strip() for c in args.categories.split(",")]

        if args.no_synthetic:
            stages = [s for s in pipeline.stages if s != "synthetic_augmentation"]
            pipeline.stages = stages

        result = pipeline.run(**pipeline_kwargs)
        examples = result["examples"]

    report = pipeline.get_report()

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {args.report}")

    print(f"\n{'=' * 60}")
    print(f"Pipeline Complete")
    print(f"Total examples: {report['stats']['total_examples']}")
    print(f"Output: {args.output}")
    print(f"Time: {report['stats'].get('total_time', 'N/A')}s")
    print(f"{'=' * 60}")

    return examples


if __name__ == "__main__":
    main()
