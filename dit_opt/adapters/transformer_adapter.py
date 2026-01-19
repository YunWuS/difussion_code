from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from dit_opt.policy import SegmentPolicy


@dataclass
class TransformerPolicyHandles:
    attach_fn: Optional[Callable[[object], None]] = None
    set_attn_scale: Optional[Callable[[float], None]] = None
    set_mlp_scale: Optional[Callable[[float], None]] = None
    set_gate_scale: Optional[Callable[[float], None]] = None
    set_cond_gain: Optional[Callable[[float], None]] = None
    set_route: Optional[Callable[[Optional[str]], None]] = None


class TransformerAdapter:
    def __init__(self, handles: Optional[TransformerPolicyHandles] = None) -> None:
        self._handles = handles or TransformerPolicyHandles()
        self._policy = SegmentPolicy()
        self._model: Optional[object] = None

    def attach(self, model: object) -> None:
        self._model = model
        if self._handles.attach_fn:
            self._handles.attach_fn(model)

    def set_policy(self, policy: SegmentPolicy) -> None:
        self._policy = policy.with_defaults()
        if self._handles.set_attn_scale:
            self._handles.set_attn_scale(self._policy.attn_scale)
        if self._handles.set_mlp_scale:
            self._handles.set_mlp_scale(self._policy.mlp_scale)
        if self._handles.set_gate_scale:
            self._handles.set_gate_scale(self._policy.gate_scale)
        if self._handles.set_cond_gain:
            self._handles.set_cond_gain(self._policy.cond_gain)
        if self._handles.set_route:
            self._handles.set_route(self._policy.route)

    def clear_policy(self) -> None:
        self.set_policy(SegmentPolicy())


class DiTTransformerAdapter(TransformerAdapter):
    """Adapter that attaches policy hooks to the DiT model implementation."""

    def __init__(self) -> None:
        super().__init__(handles=None)
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._attn_scale = 1.0
        self._mlp_scale = 1.0
        self._gate_scale = 1.0
        self._cond_gain = 1.0

    def attach(self, model: object) -> None:
        super().attach(model)
        if not hasattr(model, "blocks"):
            raise ValueError("Expected model with .blocks for DiT hook attachment.")
        self._register_hooks(model)

    def set_policy(self, policy: SegmentPolicy) -> None:
        policy = policy.with_defaults()
        if policy.route == "attn_only":
            self._attn_scale = policy.attn_scale
            self._mlp_scale = 0.0
        elif policy.route == "mlp_only":
            self._attn_scale = 0.0
            self._mlp_scale = policy.mlp_scale
        else:
            self._attn_scale = policy.attn_scale
            self._mlp_scale = policy.mlp_scale
        self._gate_scale = policy.gate_scale
        self._cond_gain = policy.cond_gain
        self._policy = policy

    def clear_policy(self) -> None:
        self.set_policy(SegmentPolicy())

    def _register_hooks(self, model: object) -> None:
        self._clear_hooks()
        for block in model.blocks:
            self._hooks.append(block.attn.register_forward_hook(self._scale_attn))
            self._hooks.append(block.mlp.register_forward_hook(self._scale_mlp))
            self._hooks.append(block.adaLN_modulation.register_forward_hook(self._scale_gate))
        if hasattr(model, "y_embedder"):
            self._hooks.append(model.y_embedder.register_forward_hook(self._scale_cond))

    def _clear_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _scale_attn(self, _module: object, _inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        if self._attn_scale == 1.0:
            return output
        return output * self._attn_scale

    def _scale_mlp(self, _module: object, _inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        if self._mlp_scale == 1.0:
            return output
        return output * self._mlp_scale

    def _scale_gate(self, _module: object, _inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        if self._gate_scale == 1.0:
            return output
        if output.shape[-1] % 6 != 0:
            return output
        hidden = output.shape[-1] // 6
        shift_msa = output[:, :hidden]
        scale_msa = output[:, hidden : 2 * hidden]
        gate_msa = output[:, 2 * hidden : 3 * hidden] * self._gate_scale
        shift_mlp = output[:, 3 * hidden : 4 * hidden]
        scale_mlp = output[:, 4 * hidden : 5 * hidden]
        gate_mlp = output[:, 5 * hidden :] * self._gate_scale
        return torch.cat([shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp], dim=1)

    def _scale_cond(self, _module: object, _inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        if self._cond_gain == 1.0:
            return output
        return output * self._cond_gain
