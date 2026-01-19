from dataclasses import dataclass
from typing import List


@dataclass
class SchedulerAdapter:
    num_steps: int
    time_max: float
    time_min: float = 0.0
    use_discrete_time: bool = True

    def make_time_sequence(self, u_schedule: List[float]) -> List[float]:
        if len(u_schedule) != self.num_steps:
            raise ValueError("u_schedule length must match num_steps")

        u_values = [min(max(u, 0.0), 1.0) for u in u_schedule]
        u_values[0] = 1.0
        u_values[-1] = 0.0
        for idx in range(1, len(u_values)):
            if u_values[idx] > u_values[idx - 1]:
                u_values[idx] = u_values[idx - 1]

        span = self.time_max - self.time_min
        time_values = [self.time_min + span * u for u in u_values]

        if self.use_discrete_time:
            time_values = [float(round(t)) for t in time_values]
            for idx in range(1, len(time_values)):
                if time_values[idx] >= time_values[idx - 1]:
                    time_values[idx] = max(time_values[idx - 1] - 1, self.time_min)
        return time_values
