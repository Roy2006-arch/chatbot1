import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cp_pipeline.schema import Problem, Platform, DifficultyLevel, DSAPattern
from cp_pipeline.classifiers.dsa_classifier import DSAClassifier


class TestDSAClassifier:
    def setup_method(self):
        self.classifier = DSAClassifier()

    def test_classify_dp(self):
        p = Problem(
            title="Knapsack Problem",
            problem_statement="Given weights and values, maximize value with DP memoization and optimal substructure",
            platform=Platform.LEETCODE,
        )
        p = self.classifier.classify(p)
        assert DSAPattern.DYNAMIC_PROGRAMMING in p.dsa_patterns

    def test_classify_binary_search(self):
        p = Problem(
            title="Binary Search",
            problem_statement="Find the target in sorted array using binary search algorithm. O(log n) bisect.",
            platform=Platform.LEETCODE,
        )
        p = self.classifier.classify(p)
        assert DSAPattern.BINARY_SEARCH in p.dsa_patterns

    def test_classify_graph(self):
        p = Problem(
            title="Shortest Path",
            problem_statement="Find shortest path in graph using Dijkstra's algorithm. Nodes and edges.",
            platform=Platform.CODEFORCES,
        )
        p = self.classifier.classify(p)
        assert DSAPattern.GRAPH in p.dsa_patterns

    def test_classify_tree(self):
        p = Problem(
            title="Tree Traversal",
            problem_statement="Binary tree inorder traversal. Node left right recursive approach.",
            platform=Platform.LEETCODE,
        )
        p = self.classifier.classify(p)
        assert DSAPattern.TREE in p.dsa_patterns

    def test_classify_two_pointers(self):
        p = Problem(
            title="Two Sum II",
            problem_statement="Find pair with given sum using two pointers technique in sorted array.",
            platform=Platform.LEETCODE,
        )
        p = self.classifier.classify(p)
        assert DSAPattern.TWO_POINTERS in p.dsa_patterns

    def test_pattern_explanation(self):
        explanation = self.classifier.explain_pattern(DSAPattern.DYNAMIC_PROGRAMMING)
        assert "subproblem" in explanation.lower() or "state" in explanation.lower()

    def test_company_frequency(self):
        freq = self.classifier.get_company_frequency(DSAPattern.ARRAY)
        assert freq == "Very High"

    def test_empty_problem(self):
        p = Problem(title="", problem_statement="", platform=Platform.LEETCODE)
        p = self.classifier.classify(p)
        assert len(p.dsa_patterns) > 0

    def test_classify_batch(self):
        problems = [
            Problem(title="DP Problem", problem_statement="dynamic programming memoization", platform=Platform.LEETCODE),
            Problem(title="Graph Problem", problem_statement="graph bfs dfs traversal", platform=Platform.CODEFORCES),
        ]
        results = self.classifier.classify_batch(problems)
        assert len(results) == 2
