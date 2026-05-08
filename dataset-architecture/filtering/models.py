from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FilterIssue:
    code: str
    message: str
    severity: Severity = Severity.MEDIUM
    dimension: str = "general"
    details: Optional[Dict[str, Any]] = None


@dataclass
class FilterResult:
    passed: bool
    score: float = 1.0
    issues: List[FilterIssue] = field(default_factory=list)
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "issues": [asdict(i) for i in self.issues],
            "dimension_scores": {k: round(v, 4) for k, v in self.dimension_scores.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def merge(cls, results: List["FilterResult"], weights: Optional[Dict[str, float]] = None) -> "FilterResult":
        if not results:
            return cls(passed=True, score=1.0)

        all_issues = []
        dim_scores = {}
        all_passed = True

        for r in results:
            all_issues.extend(r.issues)
            dim_scores.update(r.dimension_scores)
            if not r.passed:
                all_passed = False

        if weights:
            weighted = sum(
                dim_scores.get(dim, 0.0) * weights.get(dim, 0.0)
                for dim in set(list(dim_scores.keys()) + list(weights.keys()))
            )
            total_weight = sum(weights.get(dim, 0.0) for dim in dim_scores if dim in weights)
            composite = weighted / total_weight if total_weight > 0 else 0.0
        else:
            scores = [r.score for r in results]
            composite = sum(scores) / len(scores) if scores else 0.0

        return cls(
            passed=all_passed,
            score=min(1.0, max(0.0, composite)),
            issues=all_issues,
            dimension_scores=dim_scores,
        )


@dataclass
class FilterConfig:
    min_quality_score: float = 0.5
    max_hallucination_score: float = 0.3
    max_repetition_score: float = 0.4
    min_reasoning_score: float = 0.5
    max_toxicity_score: float = 0.2
    max_semantic_similarity: float = 0.85
    enable_code_validation: bool = True
    enable_markdown_validation: bool = True
    enable_hallucination_check: bool = True
    enable_repetition_check: bool = True
    enable_reasoning_check: bool = True
    enable_toxicity_check: bool = True
    enable_semantic_dedup: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    num_workers: int = 8
    output_dir: str = "filtering_data"

    @classmethod
    def from_dict(cls, d: Dict) -> "FilterConfig":
        valid_keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class FilterReport:
    total_examples: int = 0
    passed: int = 0
    rejected: int = 0
    rejection_breakdown: Dict[str, int] = field(default_factory=dict)
    dimension_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "passed": self.passed,
            "rejected": self.rejected,
            "pass_rate": round(self.passed / max(self.total_examples, 1), 4),
            "rejection_breakdown": self.rejection_breakdown,
            "dimension_stats": self.dimension_stats,
            "processing_time_seconds": round(self.processing_time, 2),
        }
