#!/usr/bin/env python3
"""
Competitive Programming Dataset Pipeline Runner

Usage:
    python run.py --leetcode data/raw/leetcode.json --codeforces data/raw/codeforces.json --output exports/cp_dataset
    python run.py --synthetic --count 1000 --output exports/cp_synthetic
    python run.py --validate-only --input data/train.jsonl
"""

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cp_pipeline.schema import Problem, Platform, Language, DifficultyLevel, DSAPattern
from cp_pipeline.orchestrator import CPPipeline
from cp_pipeline.classifiers.dsa_classifier import DSAClassifier
from cp_pipeline.curriculum.difficulty import CPDifficultyScorer
from cp_pipeline.exporters.format_converter import CPFormatConverter


def parse_args():
    parser = argparse.ArgumentParser(description="Competitive Programming Dataset Pipeline")
    parser.add_argument("--leetcode", default="", help="LeetCode data path")
    parser.add_argument("--codeforces", default="", help="Codeforces data path")
    parser.add_argument("--hackerrank", default="", help="HackerRank data path")
    parser.add_argument("--codechef", default="", help="CodeChef data path")
    parser.add_argument("--geeksforgeeks", default="", help="GeeksForGeeks data path")
    parser.add_argument("--output", default="exports/cp_dataset", help="Output directory")
    parser.add_argument("--framework", default="transformers", choices=["transformers", "axolotl", "openai"])
    parser.add_argument("--validate", action="store_true", default=True, help="Validate solution code")
    parser.add_argument("--no-validate", action="store_false", dest="validate")
    parser.add_argument("--curriculum", default="progressive", choices=["progressive", "spaced_repetition"])
    parser.add_argument("--no-tests", action="store_true", help="Skip test case generation")
    parser.add_argument("--no-debug", action="store_true", help="Skip debugging examples")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data only")
    parser.add_argument("--count", type=int, default=1000, help="Synthetic problem count")
    parser.add_argument("--report", default="", help="Save pipeline report to path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def create_synthetic_problems(count: int = 100) -> list:
    import random
    random.seed(42)

    titles = [
        "Two Sum", "Maximum Subarray", "Longest Palindromic Substring",
        "Merge Sort", "Binary Tree Traversal", "Detect Cycle in Graph",
        "Shortest Path in Grid", "Knapsack Problem", "Edit Distance",
        "Topological Sort", "Strongly Connected Components",
        "Longest Common Subsequence", "Minimum Spanning Tree",
        "Segment Tree Range Query", "Trie Autocomplete",
    ]
    patterns_list = [
        [DSAPattern.ARRAY, DSAPattern.HASH_TABLE],
        [DSAPattern.ARRAY, DSAPattern.DIVIDE_AND_CONQUER],
        [DSAPattern.STRING, DSAPattern.DYNAMIC_PROGRAMMING],
        [DSAPattern.ARRAY, DSAPattern.SORTING],
        [DSAPattern.TREE, DSAPattern.DEPTH_FIRST_SEARCH],
        [DSAPattern.GRAPH, DSAPattern.DEPTH_FIRST_SEARCH],
        [DSAPattern.GRAPH, DSAPattern.BREADTH_FIRST_SEARCH],
        [DSAPattern.DYNAMIC_PROGRAMMING, DSAPattern.MATH],
        [DSAPattern.STRING, DSAPattern.DYNAMIC_PROGRAMMING],
        [DSAPattern.GRAPH, DSAPattern.TOPOLOGICAL_SORT],
        [DSAPattern.GRAPH],
        [DSAPattern.STRING, DSAPattern.DYNAMIC_PROGRAMMING],
        [DSAPattern.GRAPH, DSAPattern.UNION_FIND],
        [DSAPattern.SEGMENT_TREE, DSAPattern.ARRAY],
        [DSAPattern.TRIE, DSAPattern.STRING],
    ]

    problems = []
    for i in range(count):
        idx = i % len(titles)
        p = Problem(
            platform=random.choice(list(Platform)),
            title=titles[idx],
            problem_statement=f"Solve the {titles[idx]} problem optimally.",
            constraints=[f"1 ≤ n ≤ 10^{random.choice([3,4,5])}"],
            difficulty=random.choice(list(DifficultyLevel)),
            dsa_patterns=patterns_list[idx] if idx < len(patterns_list) else [DSAPattern.ARRAY],
            rating=random.randint(1200, 2500),
        )
        lang = Language.PYTHON
        from .validators.code_runner import CodeValidator
        from cp_pipeline.schema import Solution
        validator = CodeValidator()
        code = f"def solve_{i}(data):\n    # Solution for {titles[idx]}\n    return result"
        p.solutions[lang.value] = Solution(
            language=lang,
            code=code,
            time_complexity=random.choice(["O(n)", "O(n log n)", "O(n²)", "O(2^n)"]),
            space_complexity=random.choice(["O(1)", "O(n)", "O(n²)"]),
            approach=f"Using {patterns_list[idx][0].value if idx < len(patterns_list) else 'array'} technique",
        )
        p.sample_test_cases.append(type("tc", (), {"input": "[1,2,3]", "expected_output": "6", "explanation": "", "is_edge_case": False, "tags": []})())
        problems.append(p)

    return problems


def main():
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.synthetic:
        from cp_pipeline.schema import Solution
        from cp_pipeline.validators.code_runner import CodeValidator

        print(f"Generating {args.count} synthetic CP problems...")
        problems = create_synthetic_problems(args.count)

        pipeline = CPPipeline()
        result = pipeline.run_synthetic_only(problems, output_dir=args.output, framework=args.framework)
        print(f"Done: {result['examples']} training examples -> {result['output_dir']}")
        return

    input_paths = {}
    if args.leetcode: input_paths[Platform.LEETCODE] = args.leetcode
    if args.codeforces: input_paths[Platform.CODEFORCES] = args.codeforces
    if args.hackerrank: input_paths[Platform.HACKERRANK] = args.hackerrank
    if args.codechef: input_paths[Platform.CODECHEF] = args.codechef
    if args.geeksforgeeks: input_paths[Platform.GEEKSFORGEEKS] = args.geeksforgeeks

    if not input_paths:
        print("No input data paths provided. Use --synthetic or provide platform data paths.")
        print("Example: python run.py --synthetic --count 5000")
        return

    pipeline = CPPipeline()
    report = pipeline.run(
        input_paths=input_paths,
        output_dir=args.output,
        validate=args.validate,
        generate_tests=not args.no_tests,
        generate_debugging=not args.no_debug,
        framework=args.framework,
        curriculum_strategy=args.curriculum,
    )

    print(f"\n{'=' * 60}")
    print(f"CP Dataset Pipeline Complete")
    print(f"Problems loaded: {report['stats']['problems_loaded']}")
    print(f"Training examples: {report['stats']['training_examples']}")
    print(f"Hard problems mined: {report['stats']['hard_problems_mined']}")
    print(f"Time: {report['stats']['pipeline_time']}s")
    print(f"Output: {report['output_dir']}")
    print(f"{'=' * 60}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved: {args.report}")


if __name__ == "__main__":
    main()
