from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class CorrectionMethod(Enum):
    AUTO_GENERATED = "auto_generated"
    HUMAN_ANNOTATED = "human_annotated"
    MODEL_REGEN = "model_regen"
    TEMPLATE_FIXED = "template_fixed"


class ExampleSource(Enum):
    FAILED_QUERY = "failed_query"
    HIGH_QUALITY = "high_quality"
    HARD_EXAMPLE = "hard_example"
    CORRECTION = "correction"
    CURATED = "curated"


@dataclass
class SelfImprovementExample:
    prompt: str
    original_response: str = ""
    corrected_response: str = ""
    source: ExampleSource = ExampleSource.FAILED_QUERY
    category: str = ""
    difficulty: int = 1
    quality_score: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)
    correction_method: CorrectionMethod = CorrectionMethod.AUTO_GENERATED
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SelfImprovementExample":
        if isinstance(d.get("source"), str):
            d["source"] = ExampleSource(d["source"])
        if isinstance(d.get("correction_method"), str):
            d["correction_method"] = CorrectionMethod(d["correction_method"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CorrectionRecord:
    failed_query_id: int
    prompt: str
    original_response: str
    corrected_response: str
    correction_method: CorrectionMethod = CorrectionMethod.AUTO_GENERATED
    validator_issues: List[str] = field(default_factory=list)
    score_before: float = 0.0
    score_after: float = 0.0
    quality_validated: bool = False
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class HardExample:
    prompt: str
    response: str = ""
    category: str = ""
    difficulty: int = 1
    failure_reasons: List[str] = field(default_factory=list)
    occurrence_count: int = 1
    cluster_id: int = 0
    embedding: Optional[List[float]] = None
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        if d.get("embedding"):
            d["embedding"] = d["embedding"][:10]
        return d


@dataclass
class EvalCase:
    prompt: str
    expected_keywords: List[str] = field(default_factory=list)
    category: str = ""
    difficulty: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelEvalResult:
    model_name: str
    run_id: str
    timestamp: str
    total_cases: int = 0
    avg_accuracy: float = 0.0
    avg_relevance: float = 0.0
    avg_coherence: float = 0.0
    avg_composite: float = 0.0
    pass_rate: float = 0.0
    grade_distribution: Dict[str, int] = field(default_factory=dict)
    failure_breakdown: Dict[str, int] = field(default_factory=dict)
    per_category: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ImprovementReport:
    run_id: str
    timestamp: str
    total_failed_queries: int = 0
    corrections_generated: int = 0
    high_quality_examples: int = 0
    hard_examples_mined: int = 0
    dataset_examples: int = 0
    model_before: Optional[ModelEvalResult] = None
    model_after: Optional[ModelEvalResult] = None
    score_improvement: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)
