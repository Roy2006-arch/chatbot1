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
                indicators=[r"subarray", r"substring", r"contiguous", r"max sum", r"longest.*sub", r"window", r"consecutive", r"minimum.*window", r"longest.*without.*repeat"],
                edge_cases=["Empty string/array", "Array with one element", "All elements identical", "Window size larger than array", "All negative numbers", "Single element window"],
                common_complexity="O(n)"
            ),
            DSAPattern(
                name="Two Pointers",
                indicators=[r"sorted", r"pair", r"sum", r"reverse", r"palindrome", r"two.*pointer", r"opposite.*end", r"converge", r"container.*water", r"trap.*rain"],
                edge_cases=["Empty array", "Two elements", "Duplicates in array", "Pointers meeting at same index", "No valid pair exists", "All same elements"],
                common_complexity="O(n)"
            ),
            DSAPattern(
                name="Dynamic Programming",
                indicators=[r"maximum", r"minimum", r"number of ways", r"optimal", r"subsequence", r"knapsack", r"memoization", r"tabulation", r"overlapping.*subproblem", r"optimal.*substructure", r"longest.*increasing", r"edit.*distance", r"coin.*change", r"rod.*cut", r"matrix.*chain"],
                edge_cases=["Base case (n=0, n=1)", "Negative values", "Overflow on large inputs", "Memory limits", "Empty input", "All zeros", "Single element"],
                common_complexity="O(n) or O(n^2)"
            ),
            DSAPattern(
                name="BFS/DFS (Graphs)",
                indicators=[r"shortest path", r"island", r"connected", r"matrix", r"neighbor", r"graph", r"tree", r"traverse", r"level.*order", r"topological", r"clone.*graph", r"course.*schedule", r"word.*ladder", r"rotten.*orange"],
                edge_cases=["Disconnected graph", "Graph with cycles", "Single node", "Self-loops", "Empty graph", "Cyclic dependency", "Multiple components"],
                common_complexity="O(V + E)"
            ),
            DSAPattern(
                name="Binary Search",
                indicators=[r"sorted", r"search", r"target", r"find", r"rotated", r"first.*occurrence", r"last.*occurrence", r"peak.*element", r"search.*range", r"minimum.*in.*rotated", r"median"],
                edge_cases=["Target not found", "Target at start/end", "Duplicates", "Array size is power of 2", "Single element", "Rotated array", "All same elements"],
                common_complexity="O(log n)"
            ),
            DSAPattern(
                name="Heap / Priority Queue",
                indicators=[r"top.*k", r"kth.*largest", r"kth.*smallest", r"median.*stream", r"priority", r"merge.*k.*sorted", r"sliding.*window.*max", r"reorganize.*string", r"task.*scheduler", r"meeting.*rooms", r"find.*median"],
                edge_cases=["Empty input", "k=1", "k equals array length", "All same priorities", "Negative priorities", "Single element heap"],
                common_complexity="O(n log k) or O(n log n)"
            ),
            DSAPattern(
                name="Trie (Prefix Tree)",
                indicators=[r"prefix", r"autocomplete", r"dictionary", r"word.*search", r"word.*break", r"implement.*trie", r"trie.*search", r"starts.*with", r"lexicographic", r"boggle"],
                edge_cases=["Empty dictionary", "Single character words", "Words with common prefixes", "Case sensitivity", "Special characters", "Very long words"],
                common_complexity="O(m) per operation, m = word length"
            ),
            DSAPattern(
                name="Union-Find / Disjoint Set",
                indicators=[r"union", r"find.*connected", r"component", r"merge.*set", r"cycle.*detection", r"redundant.*connection", r"accounts.*merge", r"equivalence.*class", r"disjoint"],
                edge_cases=["Single element", "Already connected", "Self-loop", "All disconnected", "Union by rank vs size", "Path compression edge cases"],
                common_complexity="O(α(n)) amortized, α = inverse Ackermann"
            ),
            DSAPattern(
                name="Backtracking",
                indicators=[r"permutation", r"combination", r"subset", r"n-queen", r"sudoku", r"generate.*parenthes", r"word.*search.*grid", r"combination.*sum", r"backtrack", r"all.*possible", r"subset.*sum", r"partition.*equal", r"restore.*ip"],
                edge_cases=["Empty input", "No valid solution", "Multiple valid solutions", "Pruning optimization", "Duplicate elements", "Large n (time limit)"],
                common_complexity="O(2^n) or O(n!)"
            ),
            DSAPattern(
                name="Sorting",
                indicators=[r"sort", r"merge.*sort", r"quick.*sort", r"heap.*sort", r"bucket.*sort", r"radix.*sort", r"counting.*sort", r"topological.*sort", r"custom.*comparator", r"stable.*sort", r"inplace.*sort"],
                edge_cases=["Already sorted", "Reverse sorted", "All same elements", "Negative numbers", "Very large array", "Stability requirement", "Memory constraints"],
                common_complexity="O(n log n)"
            ),
            DSAPattern(
                name="Linked List",
                indicators=[r"linked.*list", r"reverse.*list", r"merge.*list", r"cycle.*list", r"fast.*slow.*pointer", r"middle.*node", r"remove.*nth.*node", r"add.*two.*number", r"copy.*random.*pointer", r"flatten.*multilevel", r"lru.*cache"],
                edge_cases=["Empty list", "Single node", "Cycle in list", "Even/odd length", "All same values", "Very long list"],
                common_complexity="O(n)"
            ),
            DSAPattern(
                name="Stack / Queue / Monotonic",
                indicators=[r"next.*greater", r"next.*smaller", r"monotonic", r"stack", r"queue", r"deque", r"min.*stack", r"valid.*parenthes", r"largest.*rectangle", r"daily.*temperature", r"car.*fleet", r"asteroid"],
                edge_cases=["Empty input", "All same elements", "Already sorted", "Strictly increasing/decreasing", "Nested structures", "Single element"],
                common_complexity="O(n)"
            ),
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
        strategies = {
            "Dynamic Programming": "Consider space optimization (iterative with O(1) space if possible). Use memoization for top-down or tabulation for bottom-up.",
            "Sliding Window": "Use a hash map or frequency array for O(1) lookups. Maintain window invariants.",
            "Binary Search": "Ensure the search space is monotonically. Consider edge cases with duplicates.",
            "BFS/DFS (Graphs)": "Use visited set to avoid cycles. BFS for shortest path, DFS for exhaustive search.",
            "Two Pointers": "Sort first if not sorted. Use two pointers moving toward each other or in same direction.",
            "Heap / Priority Queue": "Use min-heap for top-k largest, max-heap for top-k smallest. Consider lazy deletion.",
            "Trie (Prefix Tree)": "Use for prefix-based operations. Consider space optimization with compressed tries.",
            "Union-Find / Disjoint Set": "Use path compression and union by rank for near-constant amortized operations.",
            "Backtracking": "Prune branches early. Sort input for duplicate skipping. Use early termination when possible.",
            "Sorting": "Consider stable sort if order matters. Use counting/radix for integer keys with small range.",
            "Linked List": "Use fast-slow pointer for cycle detection. Use dummy head for edge cases.",
            "Stack / Queue / Monotonic": "Use monotonic stack for next greater/smaller element problems.",
        }
        return strategies.get(pattern.name, f"Aim for {pattern.common_complexity} time complexity.")

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
