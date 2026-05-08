import json
import os
import logging
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .schema import (
    BuggyExample, SourceCode, Language, BugCategory, ErrorInfo,
    ErrorType, TestCase, DEBUG_DATASET_CONFIG, BUG_CATEGORY_DIFFICULTY,
)
from .bugs.injector import BugInjector
from .validators.code_checker import CodeChecker
from .classifiers.bug_classifier import BugClassifier
from .exporters.format_converter import DebugFormatConverter


CORRECT_CODE_BANK: Dict[Language, List[Tuple[str, str]]] = {
    Language.PYTHON: [
        ("binary_search", """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""),
        ("two_sum", """
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
"""),
        ("fibonacci", """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""),
        ("reverse_string", """
def reverse_string(s):
    chars = list(s)
    left, right = 0, len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    return "".join(chars)
"""),
        ("is_palindrome", """
def is_palindrome(s):
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
"""),
        ("max_subarray", """
def max_subarray(nums):
    max_ending = max_so_far = nums[0]
    for i in range(1, len(nums)):
        max_ending = max(nums[i], max_ending + nums[i])
        max_so_far = max(max_so_far, max_ending)
    return max_so_far
"""),
    ],
    Language.JAVASCRIPT: [
        ("binary_search", """
function binarySearch(arr, target) {
    let left = 0, right = arr.length - 1;
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
"""),
        ("two_sum", """
function twoSum(nums, target) {
    const seen = new Map();
    for (let i = 0; i < nums.length; i++) {
        const complement = target - nums[i];
        if (seen.has(complement)) {
            return [seen.get(complement), i];
        }
        seen.set(nums[i], i);
    }
    return [];
}
"""),
    ],
    Language.JAVA: [
        ("binary_search", """
public class Search {
    public static int binarySearch(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }
}
"""),
    ],
    Language.CPP: [
        ("binary_search", """
#include <vector>
using namespace std;
int binarySearch(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
"""),
    ],
}


class DebugPipeline:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DEBUG_DATASET_CONFIG
        self.injector = BugInjector()
        self.checker = CodeChecker()
        self.classifier = BugClassifier()
        self.converter = DebugFormatConverter()

        self.logger = logging.getLogger("debug_pipeline")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self.stats = {
            "code_sources": 0,
            "bugs_injected": 0,
            "examples_validated": 0,
            "examples_valid": 0,
            "training_examples": 0,
            "pipeline_time": 0.0,
        }

    def run(
        self,
        total_examples: int = 10000,
        output_dir: str = "exports/debug_dataset",
        framework: str = "transformers",
        validate: bool = True,
    ) -> Dict[str, Any]:
        start = time.time()
        self.logger.info("=" * 60)
        self.logger.info("Starting Debugging & Code Correction Pipeline")
        self.logger.info("=" * 60)

        lang_dist = self.config.get("language_distribution", {})
        code_pairs = self._build_code_pairs(total_examples, lang_dist)
        self.stats["code_sources"] = len(code_pairs)
        self.logger.info(f"Using {len(code_pairs)} code sources across languages")

        self.logger.info("Injecting bugs into correct code...")
        all_examples = self.injector.inject_batch(code_pairs, examples_per_pair=2)
        self.stats["bugs_injected"] = len(all_examples)
        self.logger.info(f"Generated {len(all_examples)} buggy examples")

        self.logger.info("Classifying bugs...")
        all_examples = self.classifier.classify_batch(all_examples)

        category_dist = {}
        for ex in all_examples:
            cat = ex.category.value
            category_dist[cat] = category_dist.get(cat, 0) + 1
        self.logger.info(f"Bug distribution: {dict(sorted(category_dist.items(), key=lambda x: -x[1])[:10])}")

        if validate:
            self.logger.info("Validating fixes...")
            validation_results = self.checker.check_batch(all_examples)
            valid = []
            for ex, result in validation_results:
                if result.passed:
                    valid.append(ex)
                else:
                    self.logger.debug(f"  Rejected {ex.id}: {result.errors[:1]}")
            self.stats["examples_validated"] = len(validation_results)
            self.stats["examples_valid"] = len(valid)
            all_examples = valid
            self.logger.info(f"Valid examples: {len(valid)}/{len(validation_results)}")

        self.logger.info("Converting to training format...")
        records = self.converter.convert_to_instructions(all_examples, include_errors=True)
        self.stats["training_examples"] = len(records)

        self.converter.export_for_finetuning(records, output_dir, framework=framework)
        self.logger.info(f"Exported {len(records)} examples to {output_dir}")

        self.stats["pipeline_time"] = round(time.time() - start, 2)

        report = {
            "status": "success",
            "stats": self.stats,
            "category_distribution": category_dist,
            "language_distribution": dict(lang_dist),
            "output_dir": output_dir,
            "completed_at": datetime.utcnow().isoformat(),
        }

        report_path = Path(output_dir) / "pipeline_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Pipeline complete: {len(records)} training examples in {self.stats['pipeline_time']}s")
        return report

    def _build_code_pairs(self, total: int, lang_dist: Dict[str, float]) -> List[Tuple[str, Language]]:
        pairs = []
        for lang_str, ratio in lang_dist.items():
            try:
                lang = Language(lang_str)
            except ValueError:
                continue

            count = int(total * ratio)
            bank = CORRECT_CODE_BANK.get(lang, [])
            if not bank:
                continue

            for _ in range(count):
                _, code = random.choice(bank)
                pairs.append((code, lang))

        random.shuffle(pairs)
        return pairs

    def run_with_custom_code(
        self,
        code_pairs: List[Tuple[str, Language, str]],
        output_dir: str = "exports/debug_custom",
        framework: str = "transformers",
    ) -> Dict[str, Any]:
        self.logger.info(f"Running with {len(code_pairs)} custom code snippets")

        all_examples = []
        for code, lang, _ in code_pairs:
            examples = self.injector.inject(code, lang, count=2)
            all_examples.extend(examples)

        all_examples = self.classifier.classify_batch(all_examples)
        validation_results = self.checker.check_batch(all_examples)
        valid = [ex for ex, result in validation_results if result.passed]

        records = self.converter.convert_to_instructions(valid)
        self.converter.export_for_finetuning(records, output_dir, framework=framework)

        return {
            "status": "success",
            "input_examples": len(code_pairs),
            "bugs_generated": len(all_examples),
            "valid_examples": len(valid),
            "training_examples": len(records),
            "output_dir": output_dir,
        }

    def get_stats(self) -> Dict:
        return self.stats
