from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SegmentPolicy:
    """Model-agnostic policy for a diffusion transformer segment."""

    attn_scale: float = 1.0
    mlp_scale: float = 1.0
    gate_scale: float = 1.0
    cond_gain: float = 1.0
    route: Optional[str] = None

    def with_defaults(self) -> "SegmentPolicy":
        return SegmentPolicy(
            attn_scale=self.attn_scale,
            mlp_scale=self.mlp_scale,
            gate_scale=self.gate_scale,
            cond_gain=self.cond_gain,
            route=self.route,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attn_scale": self.attn_scale,
            "mlp_scale": self.mlp_scale,
            "gate_scale": self.gate_scale,
            "cond_gain": self.cond_gain,
            "route": self.route,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "SegmentPolicy":
        return SegmentPolicy(
            attn_scale=float(payload.get("attn_scale", 1.0)),
            mlp_scale=float(payload.get("mlp_scale", 1.0)),
            gate_scale=float(payload.get("gate_scale", 1.0)),
            cond_gain=float(payload.get("cond_gain", 1.0)),
            route=payload.get("route"),
        )
