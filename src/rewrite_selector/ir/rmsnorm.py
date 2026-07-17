from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RMSNormWorkload:
    group_id: str
    model_id: str
    phase: str
    batch_size: int
    seq_len: int
    hidden_dim: int
    dtype: str
    seed: int
    context: str
    eps: float = 1e-6

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RMSNormWorkload":
        return cls(
            **{
                field: value[field]
                for field in cls.__dataclass_fields__
                if field in value
            }
        )


class RMSNormCandidate(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        eps: float,
        implementation: str,
        flatten: bool,
        scale_association: str,
        context: str,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.implementation = implementation
        self.flatten = flatten
        self.scale_association = scale_association
        self.context = context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.context == "norm_only":
            value = x
        elif self.context == "residual_silu":
            value = x + F.silu(x)
        else:
            raise RuntimeError(f"unsupported context: {self.context}")

        original_shape = value.shape
        if self.flatten:
            value = value.reshape(-1, self.hidden_dim)

        if self.implementation == "native":
            output = F.rms_norm(
                value,
                (self.hidden_dim,),
                self.weight,
                self.eps,
            )
        else:
            if self.implementation == "square_mul":
                variance = (value * value).mean(dim=-1, keepdim=True)
            elif self.implementation == "square_pow":
                variance = value.pow(2).mean(dim=-1, keepdim=True)
            else:
                raise RuntimeError(
                    f"unsupported implementation: {self.implementation}"
                )
            inverse_rms = torch.rsqrt(variance + self.eps)
            if self.scale_association == "left":
                output = (value * inverse_rms) * self.weight
            elif self.scale_association == "right":
                output = value * (inverse_rms * self.weight)
            else:
                raise RuntimeError(
                    "unsupported scale association: "
                    f"{self.scale_association}"
                )

        if self.flatten:
            output = output.reshape(original_shape)
        return output


def make_rmsnorm_baseline(
    workload: RMSNormWorkload,
    device: torch.device,
    dtype: torch.dtype,
) -> RMSNormCandidate:
    return RMSNormCandidate(
        workload.hidden_dim,
        workload.eps,
        implementation="native",
        flatten=False,
        scale_association="left",
        context=workload.context,
    ).to(device=device, dtype=dtype).eval()


def instantiate_rmsnorm_candidate(
    plan: dict[str, Any],
    workload: RMSNormWorkload,
    baseline: RMSNormCandidate,
    device: torch.device,
    dtype: torch.dtype,
) -> RMSNormCandidate:
    module = RMSNormCandidate(
        workload.hidden_dim,
        workload.eps,
        implementation=str(plan["implementation"]),
        flatten=bool(plan["flatten"]),
        scale_association=str(plan["scale_association"]),
        context=workload.context,
    ).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        module.weight.copy_(baseline.weight)
    return module


def make_rmsnorm_input(
    workload: RMSNormWorkload,
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
        result = torch.empty(shape, device=device, dtype=dtype)
        result.flatten()[0::2] = 4.0
        result.flatten()[1::2] = -4.0
        return result
    raise ValueError(distribution)
