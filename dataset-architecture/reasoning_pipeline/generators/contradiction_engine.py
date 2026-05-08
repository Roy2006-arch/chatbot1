import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..schema import ContradictionPair, ReasoningType, Difficulty


class ContradictionEngine:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.stats = {"generated": 0, "contradictions": 0, "consistencies": 0}

    CONTRADICTION_TEMPLATES = [
        {
            "premise": "All mammals are warm-blooded. Whales are mammals.",
            "consistent": "Therefore, whales are warm-blooded.",
            "contradictory": "Therefore, whales are cold-blooded.",
            "explanation": "The contradictory conclusion directly conflicts with the valid deduction from the premises.",
        },
        {
            "premise": "If it rains, the ground will be wet. It is raining.",
            "consistent": "Therefore, the ground is wet.",
            "contradictory": "Therefore, the ground is dry.",
            "explanation": "Modus ponens: given P→Q and P, we must conclude Q. The contradictory claims ¬Q.",
        },
        {
            "premise": "All squares are rectangles. Shape X is a square.",
            "consistent": "Therefore, shape X is a rectangle.",
            "contradictory": "Therefore, shape X is not a rectangle.",
            "explanation": "The contradictory conclusion violates the subset relationship: squares ⊆ rectangles.",
        },
        {
            "premise": "If a number is divisible by 4, it is even. 12 is divisible by 4.",
            "consistent": "Therefore, 12 is even.",
            "contradictory": "Therefore, 12 is odd.",
            "explanation": "Divisibility by 4 implies evenness. The contradictory claims an odd result, which is impossible.",
        },
        {
            "premise": "In a valid argument, if premises are true, the conclusion must be true. This argument has true premises.",
            "consistent": "Therefore, the conclusion is true.",
            "contradictory": "Therefore, the conclusion is false.",
            "explanation": "Valid argument with true premises guarantees a true conclusion. The contradictory claims a false conclusion.",
        },
    ]

    CONSISTENCY_TEMPLATES = [
        {
            "premise": "Some birds cannot fly. Penguins are birds.",
            "conclusion_a": "Some penguins might not be able to fly.",
            "conclusion_b": "All birds can fly.",
            "explanation": "Conclusion A is validly derived. Conclusion B contradicts the premise that some birds cannot fly.",
        },
        {
            "premise": "If a shape has four equal sides and four right angles, it is a square. Shape Y has four equal sides but no right angles.",
            "conclusion_a": "Shape Y is a rhombus, not a square.",
            "conclusion_b": "Shape Y is a square.",
            "explanation": "A square requires both equal sides AND right angles. Missing right angles means it's not a square.",
        },
    ]

    def generate(self, count: int = 1000, difficulty_range: range = range(1, 5)) -> List[ContradictionPair]:
        pairs = []
        for _ in range(count):
            difficulty = random.choice(list(difficulty_range))

            if random.random() < 0.6:
                pair = self._generate_contradiction(difficulty)
            else:
                pair = self._generate_consistent_pair(difficulty)

            if pair:
                pair.difficulty = Difficulty(difficulty)
                pairs.append(pair)

        self.stats["generated"] += len(pairs)
        return pairs

    def _generate_contradiction(self, difficulty: int) -> Optional[ContradictionPair]:
        template = random.choice(self.CONTRADICTION_TEMPLATES)
        return ContradictionPair(
            premise=template["premise"],
            conclusion_a=template["consistent"],
            conclusion_b=template["contradictory"],
            consistent_a=True,
            consistent_b=False,
            explanation=template["explanation"],
            reasoning_type=ReasoningType.DEDUCTIVE,
        )

    def _generate_consistent_pair(self, difficulty: int) -> Optional[ContradictionPair]:
        template = random.choice(self.CONSISTENCY_TEMPLATES)
        return ContradictionPair(
            premise=template["premise"],
            conclusion_a=template["conclusion_a"],
            conclusion_b=template["conclusion_b"],
            consistent_a=True,
            consistent_b=False,
            explanation=template["explanation"],
            reasoning_type=ReasoningType.CRITICAL,
        )

    def _generate_synthetic_pair(self, difficulty: int) -> Optional[ContradictionPair]:
        subjects = ["all humans", "every integer", "each triangle", "all algorithms", "every database"]
        pred_positive = ["are mortal", "is even", "has 180 degrees", "terminates", "supports ACID"]
        pred_negative = ["are immortal", "is odd", "has 90 degrees", "runs forever", "is eventually consistent"]
        exclude = ["Socrates", "2", "a right triangle", "quicksort", "Cassandra"]

        idx = random.randint(0, len(subjects) - 1)
        premise = f"{subjects[idx]} {pred_positive[idx]}. {exclude[idx]} is a {subjects[idx].split()[-1]}."

        return ContradictionPair(
            premise=premise,
            conclusion_a=f"Therefore, {exclude[idx]} {pred_positive[idx].split()[-1]}.",
            conclusion_b=f"Therefore, {exclude[idx]} {pred_negative[idx].split()[-1]}.",
            consistent_a=True,
            consistent_b=False,
            explanation=f"The second conclusion contradicts the universal statement that {subjects[idx]} {pred_positive[idx]}.",
            reasoning_type=ReasoningType.DEDUCTIVE,
        )

    def check_contradiction(self, statement_a: str, statement_b: str) -> Dict:
        score = 0.0
        reasons = []

        negation_pairs = [
            ("is ", "is not"), ("can ", "cannot "), ("will ", "will not "),
            ("always", "never"), ("all", "none"), ("true", "false"),
            ("valid", "invalid"), ("correct", "incorrect"), ("possible", "impossible"),
        ]

        for pos, neg in negation_pairs:
            a_has_pos = pos in (" " + statement_a.lower() + " ")
            a_has_neg = neg in (" " + statement_a.lower() + " ")
            b_has_pos = pos in (" " + statement_b.lower() + " ")
            b_has_neg = neg in (" " + statement_b.lower() + " ")
            if (a_has_pos and b_has_neg) or (a_has_neg and b_has_pos):
                score += 0.25
                reasons.append(f"'{pos}' vs '{neg}'")

        words_a = set(statement_a.lower().split())
        words_b = set(statement_b.lower().split())
        overlap = len(words_a & words_b) / max(len(words_a | words_b), 0.001)

        if overlap > 0.4 and score > 0.15:
            score += 0.2
            reasons.append(f"High overlap ({overlap:.0%}) with opposing claims")

        is_contradiction = score >= 0.3

        return {
            "contradiction_score": round(min(1.0, score), 3),
            "is_contradiction": is_contradiction,
            "reasons": reasons,
            "text_overlap": round(overlap, 3),
        }

    def generate_detection_examples(self, count: int = 500) -> List[Dict]:
        examples = []

        for pair in self.CONTRADICTION_TEMPLATES[:count]:
            examples.append({
                "premise": pair["premise"],
                "statement_a": pair["consistent"],
                "statement_b": pair["contradictory"],
                "has_contradiction": True,
                "explanation": pair["explanation"],
            })

        for pair in self.CONSISTENCY_TEMPLATES[:count]:
            examples.append({
                "premise": pair["premise"],
                "statement_a": pair["conclusion_a"],
                "statement_b": pair["conclusion_b"],
                "has_contradiction": True,
                "explanation": pair["explanation"],
            })

        for _ in range(count):
            pair = self._generate_consistent_pair(2)
            if pair:
                examples.append({
                    "premise": pair.premise,
                    "statement_a": pair.conclusion_a,
                    "statement_b": pair.conclusion_b,
                    "has_contradiction": not pair.consistent_b,
                    "explanation": pair.explanation,
                })

        return examples

    def get_stats(self) -> Dict:
        return self.stats
