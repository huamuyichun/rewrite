import torch

from rewrite_selector.ir.mlp import Workload, instantiate_candidate, make_baseline, make_input
from rewrite_selector.lowering.fingerprint import high_level_fingerprint


def test_high_level_fingerprint_is_stable_and_discriminative() -> None:
    workload = Workload("test", "tiny", "prefill", 1, 2, 8, 16, "fp32", 1)
    device = torch.device("cpu")
    baseline = make_baseline(workload, device, torch.float32)
    example = make_input(workload, device, torch.float32, 1, "normal")
    separate_plan = {"gate_up_projection": "separate", "gate_up_split": "none", "activation": "f_silu", "multiply": "out_of_place"}
    fused_plan = {"gate_up_projection": "fused", "gate_up_split": "chunk", "activation": "f_silu", "multiply": "out_of_place"}
    separate = instantiate_candidate(separate_plan, workload, baseline, device, torch.float32)
    fused = instantiate_candidate(fused_plan, workload, baseline, device, torch.float32)
    first = high_level_fingerprint(separate, example)
    second = high_level_fingerprint(separate, example)
    assert first["sha256"] == second["sha256"]
    assert first["sha256"] != high_level_fingerprint(fused, example)["sha256"]

