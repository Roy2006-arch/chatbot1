from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import hashlib
import json


class Language(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"


class BugCategory(Enum):
    SYNTAX = "syntax_error"
    OFF_BY_ONE = "off_by_one"
    NULL_POINTER = "null_pointer"
    TYPE_MISMATCH = "type_mismatch"
    LOGIC_ERROR = "logic_error"
    INFINITE_LOOP = "infinite_loop"
    DEADLOCK = "deadlock"
    RACE_CONDITION = "race_condition"
    MEMORY_LEAK = "memory_leak"
    STACK_OVERFLOW = "stack_overflow"
    INDEX_OUT_OF_BOUNDS = "index_out_of_bounds"
    DIVISION_BY_ZERO = "division_by_zero"
    UNINITIALIZED_VAR = "uninitialized_variable"
    API_MISUSE = "api_misuse"
    CONCURRENCY = "concurrency"
    PERFORMANCE = "performance"
    SECURITY = "security"
    EDGE_CASE = "edge_case"
    IMPORT_ERROR = "import_error"
    NAMING_CONFLICT = "naming_conflict"


class ErrorType(Enum):
    COMPILE_TIME = "compile_time"
    RUNTIME = "runtime"
    LOGICAL = "logical"
    WARNING = "warning"


class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SourceCode:
    language: Language
    code: str
    file_name: str = ""

    def line_count(self) -> int:
        return len(self.code.split("\n"))

    def to_dict(self) -> Dict:
        return {"language": self.language.value, "code": self.code, "file_name": self.file_name}


@dataclass
class ErrorInfo:
    error_type: ErrorType
    message: str
    line_number: int = 0
    column: int = 0
    error_code: str = ""
    stack_trace: str = ""
    category: BugCategory = BugCategory.SYNTAX

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["error_type"] = self.error_type.value
        d["category"] = self.category.value
        return d


@dataclass
class TestCase:
    input_data: str = ""
    expected_output: str = ""
    description: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BuggyExample:
    id: str = ""
    buggy_code: SourceCode = None
    corrected_code: SourceCode = None
    language: Language = Language.PYTHON
    category: BugCategory = BugCategory.LOGIC_ERROR
    error_info: Optional[ErrorInfo] = None
    severity: Severity = Severity.MEDIUM
    title: str = ""
    description: str = ""
    explanation: str = ""
    fix_strategy: str = ""
    test_cases: List[TestCase] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    difficulty: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id and self.buggy_code:
            raw = f"{self.language.value}:{self.category.value}:{self.buggy_code.code[:50]}"
            self.id = hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["language"] = self.language.value
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        if self.buggy_code:
            d["buggy_code"] = self.buggy_code.to_dict()
        if self.corrected_code:
            d["corrected_code"] = self.corrected_code.to_dict()
        if self.error_info:
            d["error_info"] = self.error_info.to_dict()
        d["test_cases"] = [tc.to_dict() for tc in self.test_cases]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "BuggyExample":
        if d.get("buggy_code"):
            d["buggy_code"] = SourceCode(**d["buggy_code"])
        if d.get("corrected_code"):
            d["corrected_code"] = SourceCode(**d["corrected_code"])
        if d.get("error_info"):
            ei = d["error_info"]
            if isinstance(ei.get("error_type"), str):
                ei["error_type"] = ErrorType(ei["error_type"])
            if isinstance(ei.get("category"), str):
                ei["category"] = BugCategory(ei["category"])
            d["error_info"] = ErrorInfo(**ei)
        if d.get("test_cases"):
            d["test_cases"] = [TestCase(**tc) for tc in d["test_cases"]]
        if isinstance(d.get("language"), str):
            d["language"] = Language(d["language"])
        if isinstance(d.get("category"), str):
            d["category"] = BugCategory(d["category"])
        if isinstance(d.get("severity"), int):
            d["severity"] = Severity(d["severity"])
        obj = cls(**{k: v for k, v in d.items() if k != "id"})
        if d.get("id"):
            obj.id = d["id"]
        return obj


@dataclass
class FixValidationResult:
    passed: bool
    compile_success: bool = False
    runtime_success: bool = False
    test_results: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


BUG_CATEGORY_DIFFICULTY = {
    BugCategory.SYNTAX: 1,
    BugCategory.OFF_BY_ONE: 2,
    BugCategory.UNINITIALIZED_VAR: 2,
    BugCategory.TYPE_MISMATCH: 2,
    BugCategory.IMPORT_ERROR: 1,
    BugCategory.NAMING_CONFLICT: 1,
    BugCategory.INDEX_OUT_OF_BOUNDS: 2,
    BugCategory.DIVISION_BY_ZERO: 2,
    BugCategory.EDGE_CASE: 2,
    BugCategory.LOGIC_ERROR: 3,
    BugCategory.INFINITE_LOOP: 3,
    BugCategory.PERFORMANCE: 3,
    BugCategory.NULL_POINTER: 3,
    BugCategory.API_MISUSE: 3,
    BugCategory.SECURITY: 4,
    BugCategory.CONCURRENCY: 4,
    BugCategory.DEADLOCK: 4,
    BugCategory.RACE_CONDITION: 4,
    BugCategory.MEMORY_LEAK: 4,
    BugCategory.STACK_OVERFLOW: 4,
}


DEBUG_DATASET_CONFIG = {
    "total_target": 300_000,
    "category_distribution": {
        "syntax_error": 0.10,
        "off_by_one": 0.08,
        "logic_error": 0.15,
        "null_pointer": 0.05,
        "type_mismatch": 0.06,
        "infinite_loop": 0.04,
        "index_out_of_bounds": 0.07,
        "division_by_zero": 0.04,
        "uninitialized_variable": 0.05,
        "edge_case": 0.12,
        "api_misuse": 0.04,
        "performance": 0.04,
        "security": 0.03,
        "concurrency": 0.03,
        "import_error": 0.03,
        "naming_conflict": 0.03,
        "other": 0.04,
    },
    "language_distribution": {
        "python": 0.40,
        "javascript": 0.25,
        "java": 0.20,
        "cpp": 0.15,
    },
    "difficulty_distribution": {
        1: 0.25,
        2: 0.35,
        3: 0.25,
        4: 0.15,
    },
    "min_explanation_length": 50,
    "max_explanation_length": 2000,
    "min_code_length": 5,
    "max_code_length": 2000,
    "test_cases_per_example": 3,
}
