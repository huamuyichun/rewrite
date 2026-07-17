import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("config_name", "family_id", "expected_count"),
    [
        (
            "mlp_bounded_v1.json",
            "mlp_gate_up_activation_control",
            19,
        ),
        (
            "rmsnorm_bounded_v1.json",
            "rmsnorm_residual_boundary",
            8,
        ),
    ],
)
def test_unified_enumeration_cli(
    tmp_path: Path,
    config_name: str,
    family_id: str,
    expected_count: int,
) -> None:
    output = tmp_path / f"{family_id}.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "enumerate_candidates.py"),
            "--config",
            str(ROOT / "configs" / "rewrites" / config_name),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    summary = json.loads(completed.stdout)
    assert result["family_id"] == summary["family_id"] == family_id
    assert len(result["candidates"]) == expected_count
    assert result["num_fx_unique_before_budget"] == expected_count
