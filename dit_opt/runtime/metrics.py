from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EvaluationMetrics:
    quality: float
    latency_ms: float
    peak_mem_mb: float
    stability: Optional[float] = None
