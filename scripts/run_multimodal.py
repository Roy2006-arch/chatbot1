#!/usr/bin/env python3
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from multimodal import MultimodalPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Dataset Pipeline")
    parser.add_argument("--config", default="", help="Path to config YAML")
    parser.add_argument("--input", default="", help="Input directory with media files")
    parser.add_argument("--output", default="multimodal_data/output", help="Output directory")
    parser.add_argument("--max-images", type=int, default=500, help="Max images to process")
    parser.add_argument("--max-screenshots", type=int, default=500, help="Max screenshots to process")
    parser.add_argument("--max-pdfs", type=int, default=100, help="Max PDFs to process")
    parser.add_argument("--analyze", action="store_true", help="Only analyze input directory, don't run pipeline")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = MultimodalPipeline(config_path=args.config if args.config else None)

    if args.analyze:
        input_dir = args.input or pipeline.config.get("multimodal", {}).get("pipeline", {}).get("input_dir", "multimodal_data/input")
        analysis = pipeline.analyze_input_directory(input_dir)
        print(f"\nInput Directory Analysis:")
        print(f"  Total files:   {analysis['total_files']}")
        print(f"  Images:        {analysis['images']}")
        print(f"  Screenshots:   {analysis['screenshots']}")
        print(f"  PDFs:          {analysis['pdfs']}")
        if analysis.get("by_extension"):
            print(f"\n  By Extension:")
            for ext, count in analysis["by_extension"].items():
                print(f"    {ext}: {count}")
        return

    report = pipeline.run(
        input_dir=args.input or None,
        output_dir=args.output,
        max_images=args.max_images,
        max_screenshots=args.max_screenshots,
        max_pdfs=args.max_pdfs,
    )

    report_path = os.path.join(args.output, "multimodal_report.json")
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
