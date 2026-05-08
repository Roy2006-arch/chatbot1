#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from filtering import FilteringPipeline
from src.pipeline.ingestion import DatasetExample


def analyze_directory(data_dir: str) -> Dict:
    pipeline = FilteringPipeline()
    all_examples = []

    for fname in os.listdir(data_dir):
        if fname.endswith((".jsonl", ".json")):
            fpath = os.path.join(data_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            all_examples.append(DatasetExample.from_dict(data))
                        except json.JSONDecodeError:
                            pass

    print(f"Loaded {len(all_examples)} examples from {data_dir}")
    report = pipeline.analyze_dataset(all_examples)
    return report


def print_report(report: Dict):
    print("\n" + "=" * 60)
    print("DATASET QUALITY ANALYSIS REPORT")
    print("=" * 60)

    print(f"\nTotal Examples: {report['total']}")
    print(f"Average Quality Score: {report['avg_quality']}")

    print(f"\nQuality Distribution:")
    for bucket, count in sorted(report["quality_distribution"].items()):
        pct = count / max(report["total"], 1) * 100
        bar = "█" * int(pct / 2)
        print(f"  {bucket}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\nCategory Distribution:")
    for cat, count in sorted(report["category_counts"].items(), key=lambda x: -x[1]):
        pct = count / max(report["total"], 1) * 100
        print(f"  {cat:30s}: {count:5d} ({pct:5.1f}%)")

    print(f"\nDimension Averages:")
    for dim, avg in sorted(report["dimension_averages"].items()):
        bar = "█" * int(avg * 50)
        print(f"  {dim:30s}: {avg:.4f} {bar}")

    print(f"\nRecommendations:")
    if report["avg_quality"] < 0.5:
        print("  ⚠ Overall quality is LOW. Consider stricter filtering thresholds.")
    if report["quality_distribution"].get("0-0.2", 0) > report["total"] * 0.1:
        print("  ⚠ More than 10% of examples have very low quality (<0.2).")
    if report["quality_distribution"].get("0.8-1.0", 0) < report["total"] * 0.2:
        print("  ⚠ Fewer than 20% of examples are high quality (>0.8). Consider sourcing better data.")
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze dataset quality")
    parser.add_argument("input_dir", help="Directory containing JSONL files")
    parser.add_argument("--output", "-o", default="", help="Save report JSON to file")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory not found: {args.input_dir}")
        sys.exit(1)

    report = analyze_directory(args.input_dir)
    print_report(report)

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
