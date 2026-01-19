from dataclasses import dataclass
import random
from typing import Callable, Iterable, List, Sequence, Tuple

from dit_opt.genotype import Genotype
from dit_opt.runtime.metrics import EvaluationMetrics
from dit_opt.policy import SegmentPolicy


@dataclass
class EvolutionConfig:
    population_size: int
    generations: int
    mutation_rate: float = 0.3
    elite_fraction: float = 0.1
    quality_threshold: float = 0.0
    objective: str = "latency"
    seed: int = 0
    attn_scales: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)
    mlp_scales: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)
    gate_scales: Sequence[float] = (0.5, 0.75, 1.0, 1.25)
    cond_gains: Sequence[float] = (0.5, 1.0, 1.5)


class EvolutionarySearch:
    def __init__(self, evaluator: Callable[[Genotype], EvaluationMetrics]) -> None:
        self._evaluator = evaluator

    def run(self, initial_population: Iterable[Genotype], config: EvolutionConfig) -> List[Genotype]:
        rng = random.Random(config.seed)
        population = list(initial_population)
        if len(population) < config.population_size:
            raise ValueError("Initial population smaller than population_size.")

        for _ in range(config.generations):
            scored = self._score_population(population, config)
            elites = self._select_elites(scored, config)
            offspring = elites[:]
            while len(offspring) < config.population_size:
                parent_a, parent_b = self._tournament(scored, rng), self._tournament(scored, rng)
                child = self._crossover(parent_a, parent_b, rng)
                if rng.random() < config.mutation_rate:
                    child = self._mutate(child, config, rng)
                offspring.append(child)
            population = offspring

        final_scored = self._score_population(population, config)
        return [genotype for genotype, _metrics, _score in final_scored]

    def _score_population(
        self, population: Sequence[Genotype], config: EvolutionConfig
    ) -> List[Tuple[Genotype, EvaluationMetrics, float]]:
        scored: List[Tuple[Genotype, EvaluationMetrics, float]] = []
        for genotype in population:
            metrics = self._evaluator(genotype)
            score = self._objective_score(metrics, config)
            scored.append((genotype, metrics, score))
        scored.sort(key=lambda item: item[2])
        return scored

    @staticmethod
    def _objective_score(metrics: EvaluationMetrics, config: EvolutionConfig) -> float:
        if metrics.quality < config.quality_threshold:
            return float("inf")
        if config.objective == "quality":
            return -metrics.quality
        if config.objective == "memory":
            return metrics.peak_mem_mb
        return metrics.latency_ms

    @staticmethod
    def _select_elites(
        scored: Sequence[Tuple[Genotype, EvaluationMetrics, float]], config: EvolutionConfig
    ) -> List[Genotype]:
        elite_count = max(1, int(round(config.population_size * config.elite_fraction)))
        return [genotype for genotype, _metrics, _score in scored[:elite_count]]

    @staticmethod
    def _tournament(
        scored: Sequence[Tuple[Genotype, EvaluationMetrics, float]], rng: random.Random
    ) -> Genotype:
        contenders = rng.sample(scored, k=2)
        contenders.sort(key=lambda item: item[2])
        return contenders[0][0]

    @staticmethod
    def _crossover(parent_a: Genotype, parent_b: Genotype, rng: random.Random) -> Genotype:
        num_steps = len(parent_a.u_schedule)
        split = rng.randint(1, max(1, num_steps - 1))
        u_schedule = parent_a.u_schedule[:split] + parent_b.u_schedule[split:]
        policies = []
        for policy_a, policy_b in zip(parent_a.policies, parent_b.policies):
            policies.append(policy_a if rng.random() < 0.5 else policy_b)
        return Genotype(
            u_schedule=u_schedule,
            segment_assignment=parent_a.segment_assignment,
            policies=policies,
        )

    @staticmethod
    def _mutate(genotype: Genotype, config: EvolutionConfig, rng: random.Random) -> Genotype:
        policies = []
        for policy in genotype.policies:
            policies.append(
                SegmentPolicy(
                    attn_scale=rng.choice(config.attn_scales),
                    mlp_scale=rng.choice(config.mlp_scales),
                    gate_scale=rng.choice(config.gate_scales),
                    cond_gain=rng.choice(config.cond_gains),
                    route=policy.route,
                )
                if rng.random() < 0.5
                else policy
            )
        u_schedule = genotype.u_schedule[:]
        if u_schedule:
            jitter_idx = rng.randrange(len(u_schedule))
            u_schedule[jitter_idx] = min(max(u_schedule[jitter_idx] + rng.uniform(-0.05, 0.05), 0.0), 1.0)
            u_schedule.sort(reverse=True)
            u_schedule[0] = 1.0
            u_schedule[-1] = 0.0
        return Genotype(
            u_schedule=u_schedule,
            segment_assignment=genotype.segment_assignment,
            policies=policies,
        )
