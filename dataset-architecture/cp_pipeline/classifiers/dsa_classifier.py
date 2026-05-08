import re
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..schema import Problem, DSAPattern, AlgorithmCategory


class DSAClassifier:
    PATTERN_SIGNATURES: Dict[DSAPattern, List[str]] = {
        DSAPattern.ARRAY: [r"array|nums?|arr|vector|index|position"],
        DSAPattern.STRING: [r"string|str|substring|palindrome|anagram|character|char"],
        DSAPattern.HASH_TABLE: [r"hash|dictionary|map|counter|defaultdict|hashmap|frequency"],
        DSAPattern.DYNAMIC_PROGRAMMING: [r"dynamic.programming|dp|memoization|memoize|tabulation|subproblem|optimal.substructure|overlapping"],
        DSAPattern.MATH: [r"math|digit|prime|factor|divisor|modulo|arithmetic|equation"],
        DSAPattern.SORTING: [r"sort|merge|quick|bubble|heap.sor?t|radix|bucket"],
        DSAPattern.GREEDY: [r"greedy|optimal.choice|locally.optimal|minimum\s*(?:number|coins|platforms)"],
        DSAPattern.DEPTH_FIRST_SEARCH: [r"depth.first|dfs|backtrack|recursive.*(?:search|traverse)|preorder|inorder|postorder"],
        DSAPattern.BINARY_SEARCH: [r"binary.search|bisect|log.*search|sorted.*search|find.*(?:peak|boundary|rotation)"],
        DSAPattern.BREADTH_FIRST_SEARCH: [r"breadth.first|bfs|level.order|shortest.*(?:path|distance).*unweighted"],
        DSAPattern.TREE: [r"tree|bst|binary.*tree|node.*(?:left|right)|root|leaf|traversal"],
        DSAPattern.MATRIX: [r"matrix|grid|2d.*array|maze|board|row.*col"],
        DSAPattern.BIT_MANIPULATION: [r"bit|^&|\||xor|shift|mask|binary.*(?:represent|digit)"],
        DSAPattern.TWO_POINTERS: [r"two\s*pointer|two.sum|three.sum|pair|opposite.*direction"],
        DSAPattern.STACK: [r"stack|monotonic.*stack|push|pop|peek|LIFO"],
        DSAPattern.HEAP: [r"heap|priority.*queue|min.*heap|max.*heap|kth.*(?:largest|smallest)"],
        DSAPattern.GRAPH: [r"graph|node|edge|directed|undirected|adjacency|topological|cycle.*detect"],
        DSAPattern.SLIDING_WINDOW: [r"sliding.*window|subarray|substring.*(?:length|max|min)|window"],
        DSAPattern.BACKTRACKING: [r"backtrack|permutation|combination|subset|n.queens|sudoku"],
        DSAPattern.DESIGN: [r"design|implement.*(?:class|data.structure)|LRU|LFU"],
        DSAPattern.TRIE: [r"trie|prefix.*tree|autocomplete|word.*dictionary"],
        DSAPattern.SEGMENT_TREE: [r"segment.*tree|range.*(?:query|update)|fenwick|BIT"],
        DSAPattern.UNION_FIND: [r"union.*find|disjoint.*set|DSU|connected.*component"],
        DSAPattern.TOPOLOGICAL_SORT: [r"topological.*sort|course.*schedule|dependency.*order"],
        DSAPattern.SHORTEST_PATH: [r"shortest.*path|dijkstra|bellman.*ford|floyd.*warshall"],
        DSAPattern.MINIMUM_SPANNING_TREE: [r"minimum.*spanning|kruskal|prim|MST"],
        DSAPattern.DIVIDE_AND_CONQUER: [r"divide.*conquer|merge.*sort|quick.*sort|binary.*search"],
        DSAPattern.RECURSION: [r"recursion|recursive|base.case|recurrence"],
        DSAPattern.MONOTONIC_STACK: [r"monotonic.*stack|next.*(?:greater|smaller)|stock.*span"],
        DSAPattern.PREFIX_SUM: [r"prefix.*sum|cumulative.*sum|range.*sum"],
        DSAPattern.INTERVAL: [r"interval|merge.*interval|overlap|meeting.*room"],
        DSAPattern.NUMBER_THEORY: [r"prime|gcd|lcm|modular|exponentiation|sieve|euclidean"],
        DSAPattern.COMBINATORICS: [r"combination|permutation|binomial|factorial|nCr|nPr"],
        DSAPattern.GAME_THEORY: [r"game.*theory|nim|minimax|sprague.*grundy|optimal.*play"],
        DSAPattern.SIMULATION: [r"simulation|simulate|process|execute"],
    }

    ALGORITHM_SIGNATURES = {
        AlgorithmCategory.DYNAMIC_PROGRAMMING: [r"dp\[|memo\[|max\(.*dp|min\(.*dp"],
        AlgorithmCategory.GRAPH: [r"graph|adj\[|visited\[|dfs|bfs|dijkstra"],
        AlgorithmCategory.SORTING: [r"\.sort\(\)|sorted\(|sort\(|Arrays\.sort|Collections\.sort"],
        AlgorithmCategory.SEARCHING: [r"binary_search|bisect|indexOf|find\(|search"],
        AlgorithmCategory.STRING_PROCESSING: [r"split|join|replace|regex|pattern|match"],
        AlgorithmCategory.NUMBER_THEORY: [r"prime|gcd|mod|pow\(|Math\.pow"],
    }

    def __init__(self):
        self.stats = {"classified": 0}

    def classify(self, problem: Problem) -> Problem:
        text = self._get_problem_text(problem).lower()

        patterns = self._detect_patterns(text)
        if not patterns:
            patterns = [DSAPattern.ARRAY]
        problem.dsa_patterns = patterns

        algorithms = self._detect_algorithms(text)
        problem.algorithm_categories = algorithms

        problem.topics = [p.value for p in patterns]

        self.stats["classified"] += 1
        return problem

    def classify_batch(self, problems: List[Problem], num_workers: int = 8) -> List[Problem]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.classify, p): p for p in problems}
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def _get_problem_text(self, problem: Problem) -> str:
        parts = [
            problem.title,
            problem.problem_statement,
            problem.solution_approach,
            problem.complexity_analysis,
            " ".join(problem.constraints),
            problem.input_format,
            problem.output_format,
            " ".join(problem.hints),
            " ".join(problem.topics),
        ]
        for sol in problem.solutions.values():
            if sol.approach:
                parts.append(sol.approach)
        return " ".join(parts)

    def _detect_patterns(self, text: str) -> List[DSAPattern]:
        scores: Dict[DSAPattern, float] = {}

        for pattern, signatures in self.PATTERN_SIGNATURES.items():
            score = 0.0
            for sig in signatures:
                matches = re.findall(sig, text)
                score += len(matches) * 0.25
            if score >= 0.25:
                scores[pattern] = score

        if not scores:
            return [DSAPattern.ARRAY]

        threshold = max(scores.values()) * 0.4
        ranked = sorted(
            [(p, s) for p, s in scores.items() if s >= threshold],
            key=lambda x: -x[1],
        )

        return [p for p, _ in ranked[:4]]

    def _detect_algorithms(self, text: str) -> List[AlgorithmCategory]:
        detected = []
        for algo, signatures in self.ALGORITHM_SIGNATURES.items():
            if any(re.search(sig, text) for sig in signatures):
                detected.append(algo)
        return detected

    def get_stats(self) -> Dict:
        return self.stats

    def explain_pattern(self, pattern: DSAPattern) -> str:
        explanations = {
            DSAPattern.DYNAMIC_PROGRAMMING: "Optimal substructure + overlapping subproblems. State definition and transition are key.",
            DSAPattern.BINARY_SEARCH: "Divide search space in half each step. Works on monotonic/ordered data.",
            DSAPattern.TWO_POINTERS: "Use two pointers moving toward each other or in same direction. O(n) time, O(1) space.",
            DSAPattern.SLIDING_WINDOW: "Expand/contract window over array/string. Track validity within window.",
            DSAPattern.BACKTRACKING: "Systematically explore all candidates, pruning invalid paths. State-space tree.",
            DSAPattern.GREEDY: "Make locally optimal choice at each step. Prove optimality via exchange argument.",
            DSAPattern.DEPTH_FIRST_SEARCH: "Explore as far as possible along each branch before backtracking.",
            DSAPattern.BREADTH_FIRST_SEARCH: "Explore all neighbors at current depth before going deeper. Shortest path in unweighted.",
            DSAPattern.UNION_FIND: "Track connected components. Union by rank + path compression for near O(1).",
            DSAPattern.TOPOLOGICAL_SORT: "Order DAG vertices so all edges go from earlier to later. Kahn's algorithm or DFS.",
            DSAPattern.SHORTEST_PATH: "Dijkstra (non-negative), Bellman-Ford (negative), Floyd-Warshall (all pairs).",
            DSAPattern.MINIMUM_SPANNING_TREE: "Connect all vertices with minimum total edge weight. Kruskal (sort+UnionFind) or Prim (heap).",
            DSAPattern.SEGMENT_TREE: "Tree over array ranges. O(log n) query/update. Lazy propagation for range updates.",
            DSAPattern.TRIE: "Tree of prefixes. O(L) search/insert. Used for autocomplete, spell check, IP routing.",
        }
        return explanations.get(pattern, "Standard DSA pattern used in competitive programming.")

    def get_company_frequency(self, pattern: DSAPattern) -> str:
        freq = {
            DSAPattern.ARRAY: "Very High",
            DSAPattern.STRING: "High",
            DSAPattern.HASH_TABLE: "Very High",
            DSAPattern.DYNAMIC_PROGRAMMING: "Very High",
            DSAPattern.TREE: "Very High",
            DSAPattern.GRAPH: "High",
            DSAPattern.BINARY_SEARCH: "High",
            DSAPattern.TWO_POINTERS: "High",
            DSAPattern.SLIDING_WINDOW: "Medium-High",
            DSAPattern.BACKTRACKING: "Medium",
            DSAPattern.HEAP: "Medium",
            DSAPattern.STACK: "Medium",
            DSAPattern.UNION_FIND: "Medium",
            DSAPattern.TRIE: "Medium",
            DSAPattern.SEGMENT_TREE: "Low",
        }
        return freq.get(pattern, "Medium")
