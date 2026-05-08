#!/usr/bin/env python3
"""
Debugging & Code Correction Dataset Pipeline Runner

Usage:
    python run.py --count 10000 --output exports/debug_dataset
    python run.py --custom --code my_code.jsonl --output exports/debug_custom
    python run.py --language python --count 5000
"""

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from debug_pipeline.orchestrator import DebugPipeline
from debug_pipeline.schema import Language, DEBUG_DATASET_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Debugging & Code Correction Dataset Pipeline")
    parser.add_argument("--count", type=int, default=10000, help="Total examples")
    parser.add_argument("--output", default="exports/debug_dataset", help="Output directory")
    parser.add_argument("--framework", default="transformers", choices=["transformers", "axolotl", "openai"])
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--language", default="", help="Language filter (python/javascript/java/cpp)")
    parser.add_argument("--custom", action="store_true", help="Use custom code from file")
    parser.add_argument("--code", default="", help="Custom code JSONL file path")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = DebugPipeline()

    if args.custom or args.code:
        if not os.path.exists(args.code):
            print(f"File not found: {args.code}")
            return

        code_pairs = []
        with open(args.code) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    lang = Language(data.get("language", "python"))
                    code = data.get("code", "")
                    title = data.get("title", "")
                    if code:
                        code_pairs.append((code, lang, title))

        print(f"Loaded {len(code_pairs)} custom code snippets")
        result = pipeline.run_with_custom_code(code_pairs, output_dir=args.output, framework=args.framework)

    else:
        lang_filter = {}
        if args.language:
            lang_filter = {args.language: 1.0}
        else:
            import copy
            lang_filter = copy.deepcopy(DEBUG_DATASET_CONFIG["language_distribution"])

        result = pipeline.run(
            total_examples=args.count,
            output_dir=args.output,
            framework=args.framework,
            validate=not args.no_validate,
        )

    print(f"\n{'=' * 60}")
    print(f"Debug Dataset Pipeline Complete")
    print(f"Bugs injected: {result.get('stats', result).get('bugs_injected', result.get('bugs_generated', 0))}")
    print(f"Valid examples: {result.get('stats', result).get('examples_valid', result.get('valid_examples', 0))}")
    print(f"Training examples: {result.get('stats', result).get('training_examples', result.get('training_examples', 0))}")
    print(f"Time: {result.get('stats', result).get('pipeline_time', 0):.2f}s")
    print(f"Output: {result['output_dir']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
