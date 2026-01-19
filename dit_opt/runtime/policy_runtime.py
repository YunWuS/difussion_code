from dataclasses import dataclass
from typing import List

from dit_opt.genotype import SegmentAssignment
from dit_opt.policy import SegmentPolicy


@dataclass
class PolicyRuntime:
    num_steps: int
    segment_assignment: SegmentAssignment
    policies: List[SegmentPolicy]

    def policy_for_step(self, step: int) -> SegmentPolicy:
        if step < 0 or step >= self.num_steps:
            raise ValueError("step must be within [0, num_steps)")
        segment = self.segment_assignment.segment_for_step(step)
        return self.policies[segment]
