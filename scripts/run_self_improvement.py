#!/usr/bin/env python3
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from self_improvement import SelfImprovementPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Improvement Dataset Generation Pipeline")
    parser.add_argument("--config", default="", help="Path to config YAML")
    parser.add_argument("--output", default="self_improvement_data", help="Output directory for dataset files")
    parser.add_argument("--max-failed", type=int, default=500, help="Max failed queries to load")
    parser.add_argument("--max-corrections", type=int, default=200, help="Max corrections to generate")
    parser.add_argument("--max-quality", type=int, default=300, help="Max high-quality examples to curate")
    parser.add_argument("--max-hard", type=int, default=100, help="Max hard examples to mine")
    parser.add_argument("--max-total", type=int, default=1000, help="Max total dataset examples")
    parser.add_argument("--analyze", action="store_true", help="Only analyze failed queries, don't run pipeline")
    parser.add_argument("--dpo-export", default="", help="Export DPO dataset to this path and exit")
    parser.add_argument("--dpo-count", type=int, default=500, help="Max examples for DPO export")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = SelfImprovementPipeline(config_path=args.config if args.config else None)

    if args.analyze:
        analysis = pipeline.analyze_failed_queries()
        print(f"\nFailed Queries Analysis:")
        print(f"  Total failed queries:  {analysis.get('total', 0)}")
        if analysis.get("by_failure_reason"):
            print(f"\n  By Failure Reason:")
            for reason, count in analysis["by_failure_reason"].items():
                print(f"    {reason}: {count}")
        if analysis.get("by_category"):
            print(f"\n  By Category:")
            for cat, count in analysis["by_category"].items():
                print(f"    {cat}: {count}")
        if "avg_score" in analysis:
            print(f"\n  Average composite score: {analysis['avg_score']}")
        return

    if args.dpo_export:
        path = pipeline.export_dpo_for_retraining(args.dpo_export, max_examples=args.dpo_count)
        print(f"DPO export saved to: {path}")
        return

    report = pipeline.run(
        output_dir=args.output,
        max_failed=args.max_failed,
        max_corrections=args.max_corrections,
        max_quality=args.max_quality,
        max_hard=args.max_hard,
        max_total=args.max_total,
    )

    report_path = os.path.join(args.output, "improvement_report.json")
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
