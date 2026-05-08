#!/usr/bin/env python3
"""
Reasoning Dataset Pipeline Runner

Usage:
    python run.py --count 10000 --output exports/reasoning_dataset
    python run.py --verify-only --input data/raw_reasoning.jsonl
    python run.py --annotation-mode --output exports/reasoning_annotation
    python run.py --contradictions --count 5000
"""

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reasoning_pipeline.orchestrator import ReasoningPipeline
from reasoning_pipeline.schema import ReasoningTask


def parse_args():
    parser = argparse.ArgumentParser(description="Reasoning Dataset Pipeline")
    parser.add_argument("--count", type=int, default=10000, help="Total examples to generate")
    parser.add_argument("--output", default="exports/reasoning_dataset", help="Output directory")
    parser.add_argument("--framework", default="transformers", choices=["transformers", "axolotl", "openai"])
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    parser.add_argument("--annotation-mode", action="store_true", help="Generate annotation guidelines")
    parser.add_argument("--verify-only", default="", help="Verify existing dataset file")
    parser.add_argument("--report", default="", help="Save pipeline report")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = ReasoningPipeline()

    if args.annotation_mode:
        result = pipeline.run_annotation_mode(output_dir=args.output)
        print(f"Annotation guidelines generated: {result['output_dir']}")
        return

    if args.verify_only:
        from reasoning_pipeline.exporters.format_converter import ReasoningFormatConverter
        from reasoning_pipeline.verifiers.reasoning_verifier import ReasoningVerifier

        verifier = ReasoningVerifier()
        converter = ReasoningFormatConverter()

        with open(args.verify_only) as f:
            raw = [json.loads(line) for line in f if line.strip()]

        print(f"Loaded {len(raw)} examples from {args.verify_only}")
        from reasoning_pipeline.schema import ReasoningExample
        examples = [ReasoningExample.from_dict(r) for r in raw]
        valid = verifier.verify_batch(examples)
        print(f"Valid: {len(valid)}, Removed: {len(examples) - len(valid)}")

        records = converter.convert_to_instructions(valid)
        out_path = args.verify_only.replace(".jsonl", "_verified.jsonl")
        converter.export_jsonl(records, out_path)
        print(f"Verified output: {out_path}")
        return

    result = pipeline.run(
        total_examples=args.count,
        output_dir=args.output,
        framework=args.framework,
        verify=not args.no_verify,
    )

    print(f"\n{'=' * 60}")
    print(f"Reasoning Dataset Pipeline Complete")
    print(f"Total generated: {result['stats']['generated']}")
    print(f"Valid after verification: {result['stats']['verified_valid']}")
    print(f"Contradiction pairs: {result['stats']['contradiction_pairs']}")
    print(f"Training examples: {result['stats']['training_examples']}")
    print(f"Time: {result['stats']['pipeline_time']}s")
    print(f"Output: {result['output_dir']}")
    print(f"{'=' * 60}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Report saved: {args.report}")


if __name__ == "__main__":
    main()
