import re
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..schema import BuggyExample, BugCategory, Severity, Language


class BugClassifier:
    PATTERN_SIGNATURES = {
        BugCategory.OFF_BY_ONE: [r"off.by.one", r"boundary", r"<=\s*\w+\s*<=", r"range\(.*[+-]1\)"],
        BugCategory.NULL_POINTER: [r"null.*pointer", r"none.*type", r"null.*dereference", r"nil"],
        BugCategory.TYPE_MISMATCH: [r"type.*mismatch", r"type.*error", r"expected.*got", r"cannot.*convert"],
        BugCategory.INFINITE_LOOP: [r"infinite.*loop", r"never.*terminat", r"endless", r"while.*true"],
        BugCategory.INDEX_OUT_OF_BOUNDS: [r"index.*out.*bound", r"out.*of.*range", r"array.*index", r"subscript"],
        BugCategory.DIVISION_BY_ZERO: [r"division.*zero", r"divide.*zero", r"zero.*division", r"zero.*divisor"],
        BugCategory.UNINITIALIZED_VAR: [r"uninitialized", r"not.*defin", r"undefined.*variable", r"before.*assignment"],
        BugCategory.MEMORY_LEAK: [r"memory.*leak", r"not.*closed", r"resource.*leak", r"never.*releas"],
        BugCategory.RACE_CONDITION: [r"race.*condition", r"data.*race", r"thread.*safe", r"concurrent.*access"],
        BugCategory.DEADLOCK: [r"deadlock", r"circular.*wait", r"hold.*wait"],
        BugCategory.SECURITY: [r"injection", r"xss", r"sql.*inject", r"buffer.*overflow", r"sanitize"],
        BugCategory.PERFORMANCE: [r"o\(n²\)", r"o\(n\^2\)", r"quadratic", r"inefficient", r"slow"],
        BugCategory.EDGE_CASE: [r"edge.*case", r"corner.*case", r"empty.*input", r"boundary.*condition"],
        BugCategory.SYNTAX: [r"syntax.*error", r"invalid.*syntax", r"parse.*error", r"unexpected.*token"],
        BugCategory.IMPORT_ERROR: [r"import.*error", r"module.*not.*found", r"cannot.*import"],
        BugCategory.NAMING_CONFLICT: [r"name.*conflict", r"shadow", r"already.*defined", r"duplicate"],
        BugCategory.STACK_OVERFLOW: [r"stack.*overflow", r"recursion.*depth", r"infinite.*recursion"],
        BugCategory.API_MISUSE: [r"api.*misuse", r"wrong.*parameter", r"invalid.*argument"],
        BugCategory.CONCURRENCY: [r"concurrency|thread.*safe|mutex|semaphore|lock"],
    }

    def __init__(self):
        self.stats = {"classified": 0}

    def classify(self, example: BuggyExample) -> BuggyExample:
        text = f"{example.title} {example.description} {example.explanation} {example.error_info.message if example.error_info else ''} {example.buggy_code.code}".lower()
        tags = []

        for category, signatures in self.PATTERN_SIGNATURES.items():
            for sig in signatures:
                if re.search(sig, text):
                    tags.append(category)

        if not tags:
            tags = [BugCategory.LOGIC_ERROR]

        if example.category == BugCategory.SYNTAX:
            tags = [BugCategory.SYNTAX]

        example.tags = list(set([t.value for t in tags]))

        if len(tags) == 1:
            example.category = tags[0]

        example.severity = self._compute_severity(example)
        self.stats["classified"] += 1
        return example

    def classify_batch(self, examples: List[BuggyExample], num_workers: int = 8) -> List[BuggyExample]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.classify, ex) for ex in examples]
            return [f.result() for f in as_completed(futures)]

    def _compute_severity(self, example: BuggyExample) -> Severity:
        severity_map = {
            BugCategory.SYNTAX: Severity.LOW,
            BugCategory.IMPORT_ERROR: Severity.LOW,
            BugCategory.NAMING_CONFLICT: Severity.LOW,
            BugCategory.OFF_BY_ONE: Severity.MEDIUM,
            BugCategory.TYPE_MISMATCH: Severity.MEDIUM,
            BugCategory.UNINITIALIZED_VAR: Severity.MEDIUM,
            BugCategory.EDGE_CASE: Severity.MEDIUM,
            BugCategory.PERFORMANCE: Severity.MEDIUM,
            BugCategory.LOGIC_ERROR: Severity.HIGH,
            BugCategory.INDEX_OUT_OF_BOUNDS: Severity.HIGH,
            BugCategory.DIVISION_BY_ZERO: Severity.HIGH,
            BugCategory.NULL_POINTER: Severity.HIGH,
            BugCategory.MEMORY_LEAK: Severity.HIGH,
            BugCategory.STACK_OVERFLOW: Severity.HIGH,
            BugCategory.INFINITE_LOOP: Severity.HIGH,
            BugCategory.API_MISUSE: Severity.HIGH,
            BugCategory.SECURITY: Severity.CRITICAL,
            BugCategory.RACE_CONDITION: Severity.CRITICAL,
            BugCategory.DEADLOCK: Severity.CRITICAL,
            BugCategory.CONCURRENCY: Severity.CRITICAL,
        }
        return severity_map.get(example.category, Severity.MEDIUM)

    def get_stats(self) -> Dict:
        return self.stats

    def get_fix_templates(self, category: BugCategory) -> List[str]:
        templates = {
            BugCategory.OFF_BY_ONE: [
                "Change '<=' to '<' in the loop condition",
                "Change range(n) to range(n-1) or range(1, n+1)",
                "Check if the loop iterates one too many or too few times",
            ],
            BugCategory.NULL_POINTER: [
                "Add a null check before accessing the object",
                "Use Optional<T> or null-conditional operator",
                "Initialize the variable before use",
            ],
            BugCategory.LOGIC_ERROR: [
                "Check the comparison operator direction",
                "Verify the condition logic (and vs or)",
                "Trace through the code with sample inputs",
            ],
            BugCategory.INDEX_OUT_OF_BOUNDS: [
                "Add bounds checking before array access",
                "Ensure index is in range [0, len-1]",
                "Account for 0-indexed vs 1-indexed confusion",
            ],
        }
        return templates.get(category, ["Review the code for correctness"])
