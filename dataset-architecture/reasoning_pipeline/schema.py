from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import hashlib
import json


class ReasoningType(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    CRITICAL = "critical"
    SYSTEMATIC = "systematic"
    COMPOSITIONAL = "compositional"
    HIERARCHICAL = "hierarchical"
    DIAGNOSTIC = "diagnostic"
    HYPOTHETICAL = "hypothetical"
    COMPARATIVE = "comparative"
    CONSTRAINT_BASED = "constraint_based"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    STRATEGIC = "strategic"
    MORAL = "moral"
    STATISTICAL = "statistical"
    ECONOMIC = "economic"


class ReasoningTask(Enum):
    MULTI_STEP = "multi_step_reasoning"
    LOGICAL_DEDUCTION = "logical_deduction"
    CONTRADICTION_DETECTION = "contradiction_detection"
    ERROR_DETECTION = "error_detection"
    CAUSE_ANALYSIS = "cause_analysis"
    STEP_VERIFICATION = "step_verification"
    DECOMPOSITION = "decomposition"
    PLANNING = "planning"
    ANALOGY_MAKING = "analogy_making"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    COUNTERFACTUAL_REASONING = "counterfactual_reasoning"
    COMPARISON_ANALYSIS = "comparison_analysis"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    OPTIMIZATION_REASONING = "optimization_reasoning"
    FORMAL_PROOF = "formal_proof"
    DEBUGGING_REASONING = "debugging_reasoning"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    RISK_ANALYSIS = "risk_analysis"
    DECISION_TREE = "decision_tree"
    TRADE_OFF_ANALYSIS = "trade_off_analysis"


class Difficulty(Enum):
    BASIC = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


@dataclass
class ReasoningStep:
    index: int
    content: str
    reasoning_type: ReasoningType
    justification: str = ""
    confidence: float = 1.0
    verification_notes: str = ""
    sub_steps: List["ReasoningStep"] = field(default_factory=list)
    is_valid: bool = True
    alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "content": self.content,
            "reasoning_type": self.reasoning_type.value,
            "justification": self.justification,
            "confidence": self.confidence,
            "verification_notes": self.verification_notes,
            "sub_steps": [s.to_dict() for s in self.sub_steps],
            "is_valid": self.is_valid,
            "alternatives": self.alternatives,
        }


@dataclass
class ReasoningExample:
    id: str = ""
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    reasoning_task: ReasoningTask = ReasoningTask.MULTI_STEP
    difficulty: Difficulty = Difficulty.INTERMEDIATE
    domain: str = ""
    question: str = ""
    context: str = ""
    correct_answer: str = ""
    wrong_answer: str = ""
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    verification: str = ""
    common_errors: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            raw = f"{self.reasoning_type.value}:{self.question[:50]}"
            self.id = hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["reasoning_type"] = self.reasoning_type.value
        d["reasoning_task"] = self.reasoning_task.value
        d["difficulty"] = self.difficulty.value
        d["reasoning_steps"] = [s.to_dict() for s in self.reasoning_steps]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ReasoningExample":
        d["reasoning_type"] = ReasoningType(d["reasoning_type"])
        d["reasoning_task"] = ReasoningTask(d["reasoning_task"])
        d["difficulty"] = Difficulty(d["difficulty"])
        d["reasoning_steps"] = [ReasoningStep(**s) for s in d.get("reasoning_steps", [])]
        return cls(**d)


@dataclass
class ContradictionPair:
    id: str = ""
    premise: str = ""
    conclusion_a: str = ""
    conclusion_b: str = ""
    consistent_a: bool = True
    consistent_b: bool = False
    explanation: str = ""
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    difficulty: Difficulty = Difficulty.INTERMEDIATE

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["reasoning_type"] = self.reasoning_type.value
        d["difficulty"] = self.difficulty.value
        return d


@dataclass
class ReasoningPlan:
    goal: str
    steps: List[str]
    dependencies: Dict[int, List[int]]
    estimated_complexity: str = ""
    verification_criteria: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


REASONING_TEMPLATES = {
    ReasoningTask.MULTI_STEP: {
        "instruction": "Work through this problem step by step, showing your reasoning at each stage.",
        "cot_format": "Let me solve this step by step.\n\n{steps}\n\nTherefore, the answer is: {answer}",
    },
    ReasoningTask.LOGICAL_DEDUCTION: {
        "instruction": "Use logical deduction to determine the correct conclusion from the given premises.",
        "cot_format": "Given the premises:\n{premises}\n\nStep 1: {step1}\nStep 2: {step2}\n...\nConclusion: {conclusion}",
    },
    ReasoningTask.CONTRADICTION_DETECTION: {
        "instruction": "Identify whether the following statements contain a contradiction. Explain why.",
        "cot_format": "Statement A: {stmt_a}\nStatement B: {stmt_b}\n\nAnalysis:\n{analysis}\n\nVerdict: {verdict}",
    },
    ReasoningTask.DECOMPOSITION: {
        "instruction": "Break down this complex problem into smaller sub-problems and solve each one.",
        "cot_format": "Problem: {problem}\n\nSub-problem 1: {sub1}\nSub-problem 2: {sub2}\n...\nCombined Solution: {solution}",
    },
    ReasoningTask.PLANNING: {
        "instruction": "Create a detailed plan to achieve the following goal, considering constraints and dependencies.",
        "cot_format": "Goal: {goal}\n\nConstraints: {constraints}\n\nPlan:\n{plan_steps}\n\nVerification: {verification}",
    },
    ReasoningTask.DEBUGGING_REASONING: {
        "instruction": "Analyze the reasoning in this solution. Find the error, explain why it's wrong, and provide the correct reasoning.",
        "cot_format": "Problem: {problem}\n\nFlawed Reasoning:\n{flawed_reasoning}\n\nError Analysis:\n{analysis}\n\nCorrect Reasoning:\n{correct_reasoning}",
    },
    ReasoningTask.MATHEMATICAL_REASONING: {
        "instruction": "Solve this mathematical problem step by step, justifying each step with the relevant mathematical principle.",
        "cot_format": "Problem: {problem}\n\nStep 1: {step1} (Reason: {reason1})\nStep 2: {step2} (Reason: {reason2})\n...\nAnswer: {answer}",
    },
    ReasoningTask.COUNTERFACTUAL_REASONING: {
        "instruction": "Analyze how the outcome would change if a key condition were different.",
        "cot_format": "Original Scenario: {scenario}\nChanged Condition: {change}\n\nIf {change}, then:\n{analysis}\n\nRevised Outcome: {outcome}",
    },
    ReasoningTask.COMPARISON_ANALYSIS: {
        "instruction": "Compare the following approaches/options, analyzing trade-offs and recommending the best choice.",
        "cot_format": "Options:\n{options}\n\nComparison Criteria: {criteria}\n\nAnalysis:\n{analysis}\n\nRecommendation: {recommendation}",
    },
}


REASONING_DOMAINS = [
    "mathematics", "programming", "physics", "logic", "philosophy",
    "economics", "biology", "chemistry", "engineering", "linguistics",
    "psychology", "politics", "ethics", "law", "medicine",
    "business", "technology", "environment", "social_science", "everyday",
]


DOMAIN_TOPICS = {
    "mathematics": ["algebra", "geometry", "calculus", "probability", "number_theory", "combinatorics", "statistics"],
    "programming": ["algorithms", "data_structures", "complexity", "debugging", "system_design", "optimization"],
    "logic": ["syllogisms", "fallacies", "truth_tables", "predicate_logic", "modal_logic"],
    "physics": ["mechanics", "thermodynamics", "electromagnetism", "quantum", "relativity"],
    "economics": ["supply_demand", "game_theory", "market_analysis", "risk_assessment"],
    "engineering": ["trade_offs", "optimization", "reliability", "design_decisions"],
}


REASONING_CONFIG = {
    "total_target": 500_000,
    "type_distribution": {
        "multi_step_reasoning": 0.15,
        "logical_deduction": 0.12,
        "contradiction_detection": 0.08,
        "error_detection": 0.08,
        "cause_analysis": 0.06,
        "step_verification": 0.06,
        "decomposition": 0.08,
        "planning": 0.06,
        "analogy_making": 0.04,
        "hypothesis_testing": 0.04,
        "counterfactual_reasoning": 0.04,
        "comparison_analysis": 0.05,
        "constraint_satisfaction": 0.03,
        "optimization_reasoning": 0.03,
        "formal_proof": 0.02,
        "debugging_reasoning": 0.04,
        "mathematical_reasoning": 0.04,
        "risk_analysis": 0.02,
        "decision_tree": 0.02,
        "trade_off_analysis": 0.02,
    },
    "difficulty_distribution": {
        "basic": 0.20,
        "intermediate": 0.40,
        "advanced": 0.30,
        "expert": 0.10,
    },
    "min_steps": 3,
    "max_steps": 15,
    "cot_required": True,
    "verification_required": True,
    "alternative_approaches": 2,
}
