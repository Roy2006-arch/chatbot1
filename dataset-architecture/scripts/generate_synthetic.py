#!/usr/bin/env python3
"""
Synthetic Data Generation Script

Generates high-quality synthetic training data across categories.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthetic.generator import SyntheticDataGenerator
from src.synthetic.augmenter import DataAugmenter
from src.quality.scoring import QualityScorer
from src.pipeline.export import DataExporter


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--categories", default="competitive_programming,algorithms_dsa,debugging,system_design,general_reasoning,math_logic,conversational_ai,technical_documentation,tool_usage,file_image_understanding", help="Comma-separated categories")
    parser.add_argument("--count", type=int, default=50000, help="Total examples to generate")
    parser.add_argument("--output", default="exports/synthetic", help="Output directory")
    parser.add_argument("--framework", default="transformers", choices=["transformers", "axolotl", "openai"])
    parser.add_argument("--augment", action="store_true", help="Apply augmentation")
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",")]
    per_category = args.count // max(len(categories), 1)

    generator = SyntheticDataGenerator()
    augmenter = DataAugmenter()
    scorer = QualityScorer()
    exporter = DataExporter()

    all_examples = []

    print(f"Generating synthetic data across {len(categories)} categories...")

    for cat in categories:
        print(f"  Generating {per_category} examples for: {cat}")
        examples = generator.generate(cat, count=per_category)
        all_examples.extend(examples)

    print(f"\nScoring {len(all_examples)} examples...")
    all_examples = scorer.score(all_examples)

    if args.augment:
        print("Applying augmentation...")
        all_examples = augmenter.augment(all_examples)

    avg_quality = sum(e.quality_score for e in all_examples) / max(len(all_examples), 1)
    print(f"Average quality score: {avg_quality:.3f}")

    print(f"Exporting to {args.output}...")
    exporter.export_for_training(all_examples, args.output, framework=args.framework)
    exporter.export_metadata(all_examples, args.output)

    print(f"\nDone! Generated {len(all_examples)} synthetic examples.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
