import json
import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .schema import Problem, CPInstructionExample, Platform, Language, DifficultyLevel, CP_DATASET_CONFIG
from .parsers.base_parser import PlatformParserFactory
from .classifiers.dsa_classifier import DSAClassifier
from .validators.code_runner import CodeValidator, TestCaseGenerator
from .curriculum.difficulty import CPDifficultyScorer, ProblemSetBuilder
from .exporters.format_converter import CPFormatConverter


class CPPipeline:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or CP_DATASET_CONFIG
        self.parser_factory = PlatformParserFactory()
        self.classifier = DSAClassifier()
        self.validator = CodeValidator()
        self.test_gen = TestCaseGenerator()
        self.difficulty_scorer = CPDifficultyScorer()
        self.builder = ProblemSetBuilder()
        self.converter = CPFormatConverter()

        self.logger = logging.getLogger("cp_pipeline")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self.stats = {
            "problems_loaded": 0,
            "problems_classified": 0,
            "solutions_validated": 0,
            "training_examples": 0,
            "hard_problems_mined": 0,
            "pipeline_time": 0.0,
        }

    def run(
        self,
        input_paths: Dict[Platform, str],
        output_dir: str = "exports/cp_dataset",
        validate: bool = True,
        generate_tests: bool = True,
        generate_debugging: bool = True,
        framework: str = "transformers",
        curriculum_strategy: str = "progressive",
    ) -> Dict[str, Any]:
        start = time.time()
        self.logger.info("=" * 60)
        self.logger.info("Starting Competitive Programming Dataset Pipeline")
        self.logger.info("=" * 60)

        problems = self._load_problems(input_paths)
        self.stats["problems_loaded"] = len(problems)
        self.logger.info(f"Loaded {len(problems)} problems")

        problems = self.classifier.classify_batch(problems)
        self.stats["problems_classified"] = len(problems)
        self.logger.info(f"Classified {len(problems)} problems into DSA patterns")

        problems = self.difficulty_scorer.score_batch(problems)
        self.logger.info("Difficulty scoring complete")

        if generate_tests:
            for problem in problems:
                edge_cases = self.test_gen.generate_edge_cases(problem)
                problem.edge_test_cases.extend(edge_cases[:5])

        if generate_debugging:
            for problem in problems:
                debugs = self.test_gen.generate_debugging_examples(problem, count=3)
                problem.debugging_examples = [d["description"] for d in debugs]

        if validate:
            validated = []
            for problem in problems:
                if problem.solutions:
                    result = self.validator.validate_solution(problem)
                    if result.passed or result.pass_rate >= 0.7:
                        validated.append(problem)
                        self.stats["solutions_validated"] += 1
                    else:
                        self.logger.debug(f"Skipping {problem.title}: {result.pass_rate:.0%} pass rate")
                else:
                    validated.append(problem)
            problems = validated
            self.logger.info(f"Validated solutions: {len(problems)} problems retained")

        curriculum = self.builder.build_curriculum(problems, strategy=curriculum_strategy)
        self.logger.info(f"Curriculum built: {sum(len(v) for v in curriculum.values())} problems across {len(curriculum)} levels")

        examples = self.converter.convert_to_instructions(
            problems,
            include_debugging=True,
            include_analysis=True,
            multi_language=True,
            chain_of_thought=True,
        )
        self.stats["training_examples"] = len(examples)
        self.logger.info(f"Generated {len(examples)} training examples")

        self.converter.export_for_finetuning(examples, output_dir, framework=framework)
        self.logger.info(f"Exported to {output_dir}")

        hard_problems = self.builder.mine_hard_problems(problems, top_k=100)
        self.stats["hard_problems_mined"] = len(hard_problems)
        self._export_hard_problems(hard_problems, output_dir)

        curriculum_dist = {dl.name: len(problems) for dl, problems in curriculum.items()}
        pattern_dist = self.builder.get_pattern_distribution(problems)

        self.stats["pipeline_time"] = round(time.time() - start, 2)

        report = {
            "status": "success",
            "stats": self.stats,
            "curriculum_distribution": curriculum_dist,
            "top_patterns": dict(sorted(pattern_dist.items(), key=lambda x: -x[1])[:15]),
            "difficulty_distribution": self._get_difficulty_distribution(problems),
            "platform_distribution": self._get_platform_distribution(problems),
            "output_dir": output_dir,
            "completed_at": datetime.utcnow().isoformat(),
        }

        report_path = Path(output_dir) / "pipeline_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Pipeline complete: {len(examples)} examples in {self.stats['pipeline_time']}s")
        self.logger.info(f"Report saved: {report_path}")

        return report

    def _load_problems(self, input_paths: Dict[Platform, str]) -> List[Problem]:
        problems = []
        for platform, path in input_paths.items():
            if not Path(path).exists():
                self.logger.warning(f"Path not found: {path}")
                continue
            self.logger.info(f"Loading {platform.value} problems from {path}")
            parser = self.parser_factory.get_parser(platform)
            try:
                platform_problems = list(parser.parse_batch(path))
                problems.extend(platform_problems)
                self.logger.info(f"  Loaded {len(platform_problems)} {platform.value} problems")
            except Exception as e:
                self.logger.error(f"  Error loading {platform.value}: {e}")
        return problems

    def _export_hard_problems(self, problems: List[Problem], output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        hard_path = output_path / "hard_problems.jsonl"
        with open(hard_path, "w", encoding="utf-8") as f:
            for p in problems:
                record = {
                    "id": p.id,
                    "title": p.title,
                    "platform": p.platform.value,
                    "difficulty": p.difficulty.value,
                    "patterns": [pat.value for pat in p.dsa_patterns],
                    "rating": p.rating,
                    "acceptance_rate": p.acceptance_rate,
                    "composite_difficulty": p.tags.get("composite_difficulty", 0),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _get_difficulty_distribution(self, problems: List[Problem]) -> Dict[str, int]:
        dist = {}
        for p in problems:
            key = p.difficulty.name
            dist[key] = dist.get(key, 0) + 1
        return dist

    def _get_platform_distribution(self, problems: List[Problem]) -> Dict[str, int]:
        dist = {}
        for p in problems:
            key = p.platform.value
            dist[key] = dist.get(key, 0) + 1
        return dist

    def run_synthetic_only(
        self,
        problems: List[Problem],
        output_dir: str = "exports/cp_synthetic",
        framework: str = "transformers",
    ) -> Dict[str, Any]:
        start = time.time()

        problems = self.classifier.classify_batch(problems)
        problems = self.difficulty_scorer.score_batch(problems)
        curriculum = self.builder.build_curriculum(problems)

        examples = self.converter.convert_to_instructions(
            problems,
            include_debugging=True,
            include_analysis=True,
            multi_language=True,
            chain_of_thought=True,
        )
        self.converter.export_for_finetuning(examples, output_dir, framework=framework)

        return {
            "status": "success",
            "problems": len(problems),
            "examples": len(examples),
            "output_dir": output_dir,
            "time": round(time.time() - start, 2),
        }


def build_default_pipeline(
    leetcode_path: str = "data/raw/leetcode",
    codeforces_path: str = "data/raw/codeforces",
    hackerrank_path: str = "data/raw/hackerrank",
    output_dir: str = "exports/cp_dataset",
) -> Dict:
    pipeline = CPPipeline()
    input_paths = {}

    if Path(leetcode_path).exists():
        input_paths[Platform.LEETCODE] = leetcode_path
    if Path(codeforces_path).exists():
        input_paths[Platform.CODEFORCES] = codeforces_path
    if Path(hackerrank_path).exists():
        input_paths[Platform.HACKERRANK] = hackerrank_path

    return pipeline.run(input_paths, output_dir=output_dir)
