from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import random

from dit_opt.policy import SegmentPolicy


@dataclass(frozen=True)
class SegmentAssignment:
    num_steps: int
    num_segments: int
    boundaries: Optional[List[int]] = None

    def segment_for_step(self, step: int) -> int:
        if step < 0 or step >= self.num_steps:
            raise ValueError("step must be within [0, num_steps)")
        if self.boundaries:
            for index, boundary in enumerate(self.boundaries):
                if step < boundary:
                    return index
            return self.num_segments - 1
        segment = int(step * self.num_segments / self.num_steps)
        return min(segment, self.num_segments - 1)


@dataclass(frozen=True)
class Genotype:
    u_schedule: List[float]
    segment_assignment: SegmentAssignment
    policies: List[SegmentPolicy]

    def policy_for_step(self, step: int) -> SegmentPolicy:
        segment = self.segment_assignment.segment_for_step(step)
        return self.policies[segment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "u_schedule": list(self.u_schedule),
            "segment_assignment": {
                "num_steps": self.segment_assignment.num_steps,
                "num_segments": self.segment_assignment.num_segments,
                "boundaries": list(self.segment_assignment.boundaries)
                if self.segment_assignment.boundaries
                else None,
            },
            "policies": [policy.to_dict() for policy in self.policies],
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "Genotype":
        assignment_payload = payload.get("segment_assignment", {})
        assignment = SegmentAssignment(
            num_steps=int(assignment_payload["num_steps"]),
            num_segments=int(assignment_payload["num_segments"]),
            boundaries=assignment_payload.get("boundaries"),
        )
        policies_payload = payload.get("policies", [])
        policies = [SegmentPolicy.from_dict(item) for item in policies_payload]
        return Genotype(
            u_schedule=[float(u) for u in payload.get("u_schedule", [])],
            segment_assignment=assignment,
            policies=policies,
        )


def random_genotype(
    num_steps: int,
    num_segments: int,
    attn_scales: List[float],
    mlp_scales: List[float],
    gate_scales: List[float],
    cond_gains: List[float],
    rng: Optional[random.Random] = None,
) -> Genotype:
    rng = rng or random.Random()
    u_values = [rng.random() for _ in range(num_steps)]
    u_values.sort(reverse=True)
    if num_steps > 0:
        u_values[0] = 1.0
        u_values[-1] = 0.0
    assignment = SegmentAssignment(num_steps=num_steps, num_segments=num_segments)
    policies = []
    for _ in range(num_segments):
        policies.append(
            SegmentPolicy(
                attn_scale=rng.choice(attn_scales),
                mlp_scale=rng.choice(mlp_scales),
                gate_scale=rng.choice(gate_scales),
                cond_gain=rng.choice(cond_gains),
            )
        )
    return Genotype(u_schedule=u_values, segment_assignment=assignment, policies=policies)
