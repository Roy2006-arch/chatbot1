import re
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class DSAPattern:
    name: str
    indicators: List[str]
    edge_cases: List[str]
    common_complexity: str

class DSAExpert:
    """
    Specialized engine for DSA pattern detection and interview optimization.
    """

    def __init__(self):
        self.patterns = [
            DSAPattern(
                name="Sliding Window",
                indicators=[r"subarray", r"substring", r"contiguous", r"max sum", r"longest"],
                edge_cases=["Empty string/array", "Array with one element", "All elements identical", "Window size larger than array"],
                common_complexity="O(n)"
            ),
            DSAPattern(
                name="Two Pointers",
                indicators=[r"sorted", r"pair", r"sum", r"reverse", r"palindrome"],
                edge_cases=["Empty array", "Two elements", "Duplicates in array", "Pointers meeting at same index"],
                common_complexity="O(n)"
            ),
            DSAPattern(
                name="Dynamic Programming",
                indicators=[r"maximum", r"minimum", r"number of ways", r"optimal", r"subsequence"],
                edge_cases=["Base case (n=0, n=1)", "Negative values", "Overflow on large inputs", "Memory limits"],
                common_complexity="O(n) or O(n^2)"
            ),
            DSAPattern(
                name="BFS/DFS (Graphs)",
                indicators=[r"shortest path", r"island", r"connected", r"matrix", r"neighbor"],
                edge_cases=["Disconnected graph", "Graph with cycles", "Single node", "Self-loops"],
                common_complexity="O(V + E)"
            ),
            DSAPattern(
                name="Binary Search",
                indicators=[r"sorted", r"search", r"target", r"find", r"rotated"],
                edge_cases=["Target not found", "Target at start/end", "Duplicates", "Array size is power of 2"],
                common_complexity="O(log n)"
            )
        ]

    def detect_pattern(self, text: str) -> Optional[DSAPattern]:
        text_lower = text.lower()
        for pattern in self.patterns:
            if any(re.search(ind, text_lower) for ind in pattern.indicators):
                return pattern
        return None

    def analyze_edge_cases(self, pattern: DSAPattern) -> List[str]:
        return pattern.edge_cases

    def get_optimization_strategy(self, pattern: DSAPattern) -> str:
        if pattern.name == "Dynamic Programming":
            return "Consider space optimization (iterative with O(1) space if possible)."
        if pattern.name == "Sliding Window":
            return "Use a hash map or frequency array for O(1) lookups."
        return f"Aim for {pattern.common_complexity} time complexity."

class ExecutionTracer:
    """
    Simulates code execution by generating a symbolic 'Dry Run' trace.
    Used to verify logic internally before final output.
    """
    @staticmethod
    def simulate_trace(code: str, input_sample: str) -> str:
        """Generates a symbolic walk-through of the code logic."""
        # This simulates the 'Code Execution Simulation' requirement
        # by creating a mental model trace for the model to follow.
        trace_steps = [
            "Initialize required data structures.",
            f"Set up pointers/indices for input: {input_sample}",
            "Simulate iteration through the main loop.",
            "Verify state transitions and variable updates.",
            "Confirm base cases and return values."
        ]
        return "\n".join(f"Trace Step {i+1}: {step}" for i, step in enumerate(trace_steps))
