from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from enum import Enum
import hashlib
import json


class Platform(Enum):
    LEETCODE = "leetcode"
    HACKERRANK = "hackerrank"
    CODEFORCES = "codeforces"
    CODECHEF = "codechef"
    GEEKSFORGEEKS = "geeksforgeeks"


class DifficultyLevel(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


class DSAPattern(Enum):
    ARRAY = "array"
    STRING = "string"
    HASH_TABLE = "hash_table"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    MATH = "math"
    SORTING = "sorting"
    GREEDY = "greedy"
    DEPTH_FIRST_SEARCH = "depth_first_search"
    BINARY_SEARCH = "binary_search"
    BREADTH_FIRST_SEARCH = "breadth_first_search"
    TREE = "tree"
    MATRIX = "matrix"
    BIT_MANIPULATION = "bit_manipulation"
    TWO_POINTERS = "two_pointers"
    STACK = "stack"
    HEAP = "heap"
    GRAPH = "graph"
    SLIDING_WINDOW = "sliding_window"
    BACKTRACKING = "backtracking"
    DESIGN = "design"
    TRIE = "trie"
    SEGMENT_TREE = "segment_tree"
    UNION_FIND = "union_find"
    TOPOLOGICAL_SORT = "topological_sort"
    SHORTEST_PATH = "shortest_path"
    MINIMUM_SPANNING_TREE = "minimum_spanning_tree"
    DIVIDE_AND_CONQUER = "divide_and_conquer"
    RECURSION = "recursion"
    MONOTONIC_STACK = "monotonic_stack"
    PREFIX_SUM = "prefix_sum"
    INTERVAL = "interval"
    BRAINTEASER = "brainteaser"
    RESERVOIR_SAMPLING = "reservoir_sampling"
    REJECTION_SAMPLING = "rejection_sampling"
    GEOMETRY = "geometry"
    COMBINATORICS = "combinatorics"
    NUMBER_THEORY = "number_theory"
    PROBABILITY = "probability"
    GAME_THEORY = "game_theory"
    SIMULATION = "simulation"
    DATA_STREAM = "data_stream"
    ITERATOR = "iterator"
    CONCURRENCY = "concurrency"
    DATABASE = "database"
    SHELL = "shell"


class AlgorithmCategory(Enum):
    SORTING = "sorting"
    SEARCHING = "searching"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    GRAPH = "graph"
    TREE = "tree"
    STRING_PROCESSING = "string_processing"
    MATH = "math"
    GEOMETRY = "geometry"
    COMBINATORICS = "combinatorics"
    NUMBER_THEORY = "number_theory"
    GAME_THEORY = "game_theory"
    PROBABILITY = "probability"
    OPTIMIZATION = "optimization"
    DATA_STRUCTURE = "data_structure"
    CONCURRENCY = "concurrency"


class Language(Enum):
    PYTHON = "python"
    JAVA = "java"
    CPP = "cpp"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    KOTLIN = "kotlin"
    SWIFT = "swift"


@dataclass
class TestCase:
    input: str
    expected_output: str
    explanation: Optional[str] = None
    is_edge_case: bool = False
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Solution:
    language: Language
    code: str
    time_complexity: str
    space_complexity: str
    approach: str
    approach_type: str = ""  # "optimal", "brute_force", "divide_and_conquer", etc.
    runtime_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    is_optimized: bool = True
    line_count: int = 0

    def __post_init__(self):
        self.line_count = len(self.code.strip().split("\n"))

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Problem:
    id: str = ""
    title: str = ""
    platform: Platform = Platform.LEETCODE
    platform_id: str = ""
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    dsa_patterns: List[DSAPattern] = field(default_factory=list)
    algorithm_categories: List[AlgorithmCategory] = field(default_factory=list)
    problem_statement: str = ""
    constraints: List[str] = field(default_factory=list)
    input_format: str = ""
    output_format: str = ""
    sample_test_cases: List[TestCase] = field(default_factory=list)
    hidden_test_cases: List[TestCase] = field(default_factory=list)
    edge_test_cases: List[TestCase] = field(default_factory=list)
    solutions: Dict[str, Solution] = field(default_factory=dict)
    hints: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    similar_problems: List[str] = field(default_factory=list)
    solution_approach: str = ""
    complexity_analysis: str = ""
    common_mistakes: List[str] = field(default_factory=list)
    debugging_examples: List[str] = field(default_factory=list)
    acceptance_rate: float = 0.0
    frequency: float = 0.0
    rating: int = 0
    tags: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            raw = f"{self.platform.value}:{self.platform_id}:{self.title}"
            self.id = hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["platform"] = self.platform.value
        d["difficulty"] = self.difficulty.value
        d["dsa_patterns"] = [p.value for p in self.dsa_patterns]
        d["algorithm_categories"] = [c.value for c in self.algorithm_categories]
        d["sample_test_cases"] = [tc.to_dict() for tc in self.sample_test_cases]
        d["hidden_test_cases"] = [tc.to_dict() for tc in self.hidden_test_cases]
        d["edge_test_cases"] = [tc.to_dict() for tc in self.edge_test_cases]
        d["solutions"] = {k: v.to_dict() for k, v in self.solutions.items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Problem":
        d["platform"] = Platform(d["platform"])
        d["difficulty"] = DifficultyLevel(d["difficulty"]) if isinstance(d.get("difficulty"), int) else DifficultyLevel(d["difficulty"])
        d["dsa_patterns"] = [DSAPattern(p) for p in d.get("dsa_patterns", [])]
        d["algorithm_categories"] = [AlgorithmCategory(c) for c in d.get("algorithm_categories", [])]
        d["sample_test_cases"] = [TestCase(**tc) if not isinstance(tc, TestCase) else tc for tc in d.get("sample_test_cases", [])]
        d["hidden_test_cases"] = [TestCase(**tc) if not isinstance(tc, TestCase) else tc for tc in d.get("hidden_test_cases", [])]
        d["edge_test_cases"] = [TestCase(**tc) if not isinstance(tc, TestCase) else tc for tc in d.get("edge_test_cases", [])]
        if "solutions" in d and isinstance(d["solutions"], dict):
            solutions = {}
            for lang_key, sol_data in d["solutions"].items():
                if isinstance(sol_data, dict):
                    sol_data["language"] = Language(sol_data["language"]) if isinstance(sol_data.get("language"), str) else sol_data.get("language", Language.PYTHON)
                    solutions[lang_key] = Solution(**sol_data)
                else:
                    solutions[lang_key] = sol_data
            d["solutions"] = solutions
        return cls(**d)


@dataclass
class CPInstructionExample:
    instruction: str
    input: str
    output: str
    problem: Optional[Problem] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
        }


INSTRUCTION_TEMPLATES = {
    "solve": "Solve the following competitive programming problem. Provide an optimized solution with complexity analysis.",
    "explain": "Explain the approach to solve this problem step by step, then provide the implementation.",
    "debug": "The following solution has a bug. Identify the issue, explain why it's wrong, and provide the corrected version.",
    "optimize": "The current solution has {current_complexity}. Optimize it to achieve {target_complexity}.",
    "complexity": "Analyze the time and space complexity of this solution and explain the trade-offs.",
    "edge_cases": "List all edge cases for this problem and explain how to handle each one.",
    "pattern": "Which DSA pattern(s) does this problem use? Explain why and how to apply them.",
    "compare": "Compare different approaches to solve this problem (brute force vs optimal).",
    "test": "Write comprehensive test cases for this problem covering all edge cases.",
    "multi_lang": "Provide solutions in Python, Java, and C++ for this problem.",
}


CP_DATASET_CONFIG = {
    "platoforms": ["leetcode", "hackerrank", "codeforces", "codechef", "geeksforgeeks"],
    "difficulty_distribution": {
        "easy": 0.20,
        "medium": 0.40,
        "hard": 0.30,
        "expert": 0.10,
    },
    "pattern_coverage": {
        "dynamic_programming": 0.08,
        "array": 0.08,
        "string": 0.06,
        "hash_table": 0.06,
        "depth_first_search": 0.05,
        "tree": 0.05,
        "binary_search": 0.04,
        "graph": 0.04,
        "greedy": 0.04,
        "two_pointers": 0.03,
        "stack": 0.03,
        "heap": 0.03,
        "backtracking": 0.03,
        "sliding_window": 0.02,
        "bit_manipulation": 0.02,
        "union_find": 0.02,
        "segment_tree": 0.02,
        "trie": 0.02,
        "monotonic_stack": 0.02,
        "prefix_sum": 0.02,
        "divide_and_conquer": 0.02,
        "other": 0.22,
    },
    "languages": ["python", "java", "cpp", "javascript"],
    "min_test_cases_per_problem": 5,
    "max_test_cases_per_problem": 30,
    "synthetic_debugging_examples_per_problem": 3,
}
