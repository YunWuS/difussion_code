from dataclasses import dataclass
from typing import Iterable, List, Sequence

from dit_opt.genotype import Genotype
from dit_opt.runtime.metrics import EvaluationMetrics


@dataclass(frozen=True)
class ProxyEvaluatorConfig:
    base_quality: float = 1.0
    base_latency_ms: float = 100.0
    base_mem_mb: float = 1000.0


class ProxyEvaluator:
    """Deterministic proxy evaluator for quality/latency/memory tradeoffs."""

    def __init__(self, config: ProxyEvaluatorConfig) -> None:
        self._config = config

    def evaluate(self, genotype: Genotype) -> EvaluationMetrics:
        per_step_policies = self._policies_for_steps(genotype)
        avg_attn = self._average([p.attn_scale for p in per_step_policies])
        avg_mlp = self._average([p.mlp_scale for p in per_step_policies])
        avg_gate = self._average([p.gate_scale for p in per_step_policies])
        avg_cond = self._average([p.cond_gain for p in per_step_policies])

        quality_multiplier = max(0.0, (0.5 * avg_attn + 0.5 * avg_mlp)) * avg_gate * avg_cond
        latency_multiplier = max(0.05, 0.55 * avg_attn + 0.45 * avg_mlp)
        mem_multiplier = max(0.05, 0.35 * avg_attn + 0.65 * avg_mlp)

        return EvaluationMetrics(
            quality=self._config.base_quality * quality_multiplier,
            latency_ms=self._config.base_latency_ms * latency_multiplier,
            peak_mem_mb=self._config.base_mem_mb * mem_multiplier,
            stability=None,
        )

    def batch_evaluate(self, genotypes: Iterable[Genotype]) -> List[EvaluationMetrics]:
        return [self.evaluate(genotype) for genotype in genotypes]

    def _policies_for_steps(self, genotype: Genotype) -> Sequence:
        num_steps = len(genotype.u_schedule)
        return [genotype.policy_for_step(step) for step in range(num_steps)]

    @staticmethod
    def _average(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)
