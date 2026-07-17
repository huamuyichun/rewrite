from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Workload:
    group_id: str
    model_id: str
    phase: str
    batch_size: int
    seq_len: int
    hidden_dim: int
    intermediate_dim: int
    dtype: str
    seed: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Workload":
        return cls(
            **{
                field: value[field]
                for field in cls.__dataclass_fields__
            }
        )


class SeparateMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation: str,
        multiply: str,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.activation = activation
        self.multiply = multiply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden_gate = _activate(gate, self.activation)
        hidden = _multiply(hidden_gate, up, self.multiply)
        return self.down_proj(hidden)


class FusedGateUpMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        split_mode: str,
        packing_order: str,
        activation: str,
        multiply: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            hidden_dim,
            2 * intermediate_dim,
            bias=False,
        )
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.intermediate_dim = intermediate_dim
        self.split_mode = split_mode
        self.packing_order = packing_order
        self.activation = activation
        self.multiply = multiply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = self.gate_up_proj(x)
        if self.split_mode == "chunk":
            first, second = torch.chunk(packed, 2, dim=-1)
        elif self.split_mode == "split":
            first, second = torch.split(
                packed,
                self.intermediate_dim,
                dim=-1,
            )
        elif self.split_mode == "narrow":
            first = torch.narrow(
                packed,
                -1,
                0,
                self.intermediate_dim,
            )
            second = torch.narrow(
                packed,
                -1,
                self.intermediate_dim,
                self.intermediate_dim,
            )
        else:
            raise RuntimeError(f"unsupported split_mode: {self.split_mode}")

        if self.packing_order == "gate_up":
            gate, up = first, second
        elif self.packing_order == "up_gate":
            up, gate = first, second
        else:
            raise RuntimeError(
                f"unsupported packing_order: {self.packing_order}"
            )
        hidden_gate = _activate(gate, self.activation)
        hidden = _multiply(hidden_gate, up, self.multiply)
        return self.down_proj(hidden)


def _activate(value: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "f_silu":
        return F.silu(value)
    if activation == "manual_silu":
        return value * torch.sigmoid(value)
    raise RuntimeError(f"unsupported activation: {activation}")


def _multiply(
    left: torch.Tensor,
    right: torch.Tensor,
    multiply: str,
) -> torch.Tensor:
    if multiply == "out_of_place":
        return left * right
    if multiply == "inplace":
        return left.mul_(right)
    raise RuntimeError(f"unsupported multiply: {multiply}")


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_baseline(
    workload: Workload,
    device: torch.device,
    dtype: torch.dtype,
) -> SeparateMLP:
    return SeparateMLP(
        workload.hidden_dim,
        workload.intermediate_dim,
        activation="f_silu",
        multiply="out_of_place",
    ).to(device=device, dtype=dtype).eval()


def instantiate_candidate(
    plan: dict[str, Any],
    workload: Workload,
    baseline: SeparateMLP,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    projection = str(plan["gate_up_projection"])
    activation = str(plan.get("activation", "f_silu"))
    multiply = str(plan.get("multiply", "out_of_place"))
    if projection == "separate":
        model: nn.Module = SeparateMLP(
            workload.hidden_dim,
            workload.intermediate_dim,
            activation,
            multiply,
        )
    elif projection in {"fused", "merged"}:
        model = FusedGateUpMLP(
            workload.hidden_dim,
            workload.intermediate_dim,
            str(plan.get("gate_up_split", "chunk")),
            str(plan.get("packing_order", "gate_up")),
            activation,
            multiply,
        )
    else:
        raise ValueError(f"unsupported projection: {projection}")

    model.to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        if isinstance(model, SeparateMLP):
            model.gate_proj.weight.copy_(baseline.gate_proj.weight)
            model.up_proj.weight.copy_(baseline.up_proj.weight)
        else:
            packing_order = str(plan.get("packing_order", "gate_up"))
            weights = (
                [baseline.gate_proj.weight, baseline.up_proj.weight]
                if packing_order == "gate_up"
                else [baseline.up_proj.weight, baseline.gate_proj.weight]
            )
            model.gate_up_proj.weight.copy_(torch.cat(weights, dim=0))
        model.down_proj.weight.copy_(baseline.down_proj.weight)
    return model


def make_input(
    workload: Workload,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    distribution: str,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (
        workload.batch_size,
        workload.seq_len,
        workload.hidden_dim,
    )
    if distribution == "normal":
        return torch.randn(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    if distribution == "uniform":
        return (
            torch.rand(
                shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * 2
            - 1
        )
    if distribution == "zeros":
        return torch.zeros(shape, device=device, dtype=dtype)
    if distribution == "extremes":
        values = torch.empty(shape, device=device, dtype=dtype)
        values.flatten()[0::2] = 4.0
        values.flatten()[1::2] = -4.0
        return values
    raise ValueError(f"unsupported input distribution: {distribution}")
