from dataclasses import dataclass
import random
from typing import Callable, Iterable, List, Sequence, Tuple

from dit_opt.genotype import Genotype
from dit_opt.runtime.metrics import EvaluationMetrics
from dit_opt.policy import SegmentPolicy


@dataclass
class NSGA2Config:
    population_size: int
    generations: int
    mutation_rate: float = 0.3
    seed: int = 0
    attn_scales: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)
    mlp_scales: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)
    gate_scales: Sequence[float] = (0.5, 0.75, 1.0, 1.25)
    cond_gains: Sequence[float] = (0.5, 1.0, 1.5)


class NSGA2Search:
    def __init__(self, evaluator: Callable[[Genotype], EvaluationMetrics]) -> None:
        self._evaluator = evaluator

    def run(self, initial_population: Iterable[Genotype], config: NSGA2Config) -> List[Genotype]:
        rng = random.Random(config.seed)
        population = list(initial_population)
        if len(population) < config.population_size:
            raise ValueError("Initial population smaller than population_size.")

        for _ in range(config.generations):
            combined = population[:]
            while len(combined) < 2 * config.population_size:
                parent_a, parent_b = rng.sample(population, 2)
                child = self._crossover(parent_a, parent_b, rng)
                if rng.random() < config.mutation_rate:
                    child = self._mutate(child, config, rng)
                combined.append(child)
            scored = self._score_population(combined)
            fronts = self._fast_non_dominated_sort(scored)
            population = self._select_next_population(fronts, config.population_size)

        return [genotype for genotype, _metrics in self._score_population(population)]

    def _score_population(self, population: Sequence[Genotype]) -> List[Tuple[Genotype, EvaluationMetrics]]:
        return [(genotype, self._evaluator(genotype)) for genotype in population]

    @staticmethod
    def _fast_non_dominated_sort(
        scored: Sequence[Tuple[Genotype, EvaluationMetrics]]
    ) -> List[List[Tuple[Genotype, EvaluationMetrics]]]:
        fronts: List[List[Tuple[Genotype, EvaluationMetrics]]] = []
        domination_counts = [0 for _ in scored]
        dominated_sets: List[List[int]] = [[] for _ in scored]
        first_front: List[int] = []
        for i, individual in enumerate(scored):
            for j, other in enumerate(scored):
                if i == j:
                    continue
                if NSGA2Search._dominates(individual[1], other[1]):
                    dominated_sets[i].append(j)
                elif NSGA2Search._dominates(other[1], individual[1]):
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                first_front.append(i)
        current_front = first_front
        while current_front:
            fronts.append([scored[idx] for idx in current_front])
            next_front: List[int] = []
            for idx in current_front:
                for dominated_idx in dominated_sets[idx]:
                    domination_counts[dominated_idx] -= 1
                    if domination_counts[dominated_idx] == 0:
                        next_front.append(dominated_idx)
            current_front = next_front
        return fronts

    @staticmethod
    def _dominates(left: EvaluationMetrics, right: EvaluationMetrics) -> bool:
        better_or_equal = (
            left.quality >= right.quality
            and left.latency_ms <= right.latency_ms
            and left.peak_mem_mb <= right.peak_mem_mb
        )
        strictly_better = (
            left.quality > right.quality
            or left.latency_ms < right.latency_ms
            or left.peak_mem_mb < right.peak_mem_mb
        )
        return better_or_equal and strictly_better

    @staticmethod
    def _crowding_distance(front: Sequence[Tuple[Genotype, EvaluationMetrics]]) -> List[float]:
        if not front:
            return []
        distance = [0.0 for _ in front]
        objectives = [
            ("quality", True),
            ("latency_ms", False),
            ("peak_mem_mb", False),
        ]
        for objective, maximize in objectives:
            values = [getattr(item[1], objective) for item in front]
            sorted_indices = sorted(range(len(front)), key=lambda idx: values[idx], reverse=maximize)
            min_val = values[sorted_indices[-1]]
            max_val = values[sorted_indices[0]]
            distance[sorted_indices[0]] = float("inf")
            distance[sorted_indices[-1]] = float("inf")
            if max_val == min_val:
                continue
            for i in range(1, len(front) - 1):
                prev_val = values[sorted_indices[i - 1]]
                next_val = values[sorted_indices[i + 1]]
                distance[sorted_indices[i]] += (next_val - prev_val) / (max_val - min_val)
        return distance

    @staticmethod
    def _select_next_population(
        fronts: Sequence[Sequence[Tuple[Genotype, EvaluationMetrics]]], population_size: int
    ) -> List[Genotype]:
        selected: List[Genotype] = []
        for front in fronts:
            if len(selected) + len(front) <= population_size:
                selected.extend([genotype for genotype, _metrics in front])
                continue
            distances = NSGA2Search._crowding_distance(front)
            ranked = sorted(range(len(front)), key=lambda idx: distances[idx], reverse=True)
            remaining = population_size - len(selected)
            for idx in ranked[:remaining]:
                selected.append(front[idx][0])
            break
        return selected

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
    def _mutate(genotype: Genotype, config: NSGA2Config, rng: random.Random) -> Genotype:
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
