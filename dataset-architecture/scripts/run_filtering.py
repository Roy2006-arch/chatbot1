#!/usr/bin/env python3
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from filtering import FilteringPipeline
from src.pipeline.ingestion import DatasetExample


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Dataset Quality Filtering Pipeline")
    parser.add_argument("--input", required=True, help="Input JSONL file with dataset examples")
    parser.add_argument("--output", default="filtering_data", help="Output directory")
    parser.add_argument("--config", default="", help="Path to config YAML")
    parser.add_argument("--stages", nargs="+", default=None,
                        choices=["toxicity", "hallucination", "repetition", "reasoning", "code", "markdown", "quality", "semantic", "rank"],
                        help="Filtering stages to run (default: all)")
    parser.add_argument("--analyze", action="store_true", help="Only analyze dataset, don't filter")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to process (0 = all)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    examples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    examples.append(DatasetExample.from_dict(data))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")

    if args.max_samples > 0:
        examples = examples[:args.max_samples]

    print(f"Loaded {len(examples)} examples from {args.input}")

    pipeline = FilteringPipeline(config_path=args.config if args.config else None)

    if args.analyze:
        report = pipeline.analyze_dataset(examples)
        print(f"\nDataset Analysis:")
        print(f"  Total examples: {report['total']}")
        print(f"  Average quality score: {report['avg_quality']}")
        print(f"\n  Quality Distribution:")
        for bucket, count in report["quality_distribution"].items():
            print(f"    {bucket}: {count} ({count/max(report['total'],1)*100:.1f}%)")
        print(f"\n  Category Counts:")
        for cat, count in sorted(report["category_counts"].items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")
        print(f"\n  Dimension Averages:")
        for dim, avg in sorted(report["dimension_averages"].items()):
            print(f"    {dim}: {avg}")
        report_path = os.path.join(args.output, "analysis_report.json")
        os.makedirs(args.output, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {report_path}")
        return

    print(f"Running filtering stages: {args.stages or 'all'}")
    filtered = pipeline.run(examples, stages=args.stages, output_dir=args.output)

    report = pipeline.get_report()
    print(f"\nFiltering Complete:")
    print(f"  Total: {report.total_examples}")
    print(f"  Passed: {report.passed}")
    print(f"  Rejected: {report.rejected}")
    print(f"  Pass rate: {report.passed/max(report.total_examples,1)*100:.1f}%")
    print(f"  Time: {report.processing_time:.2f}s")
    print(f"\n  Rejection Breakdown:")
    for stage, count in report.rejection_breakdown.items():
        if count > 0:
            print(f"    {stage}: {count}")
    print(f"\nOutput saved to {args.output}")


if __name__ == "__main__":
    main()
