import json
from pathlib import Path

from scripts.check_migration import (
    PHASE1_SESSIONS,
    _check_phase1_artifacts,
    _check_qwen,
)


def test_qwen_check_validates_shape_and_weights(tmp_path: Path) -> None:
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 3584,
                "intermediate_size": 18944,
            }
        )
    )
    (model_dir / "model-00001-of-00004.safetensors").write_bytes(b"test")

    ok, details = _check_qwen(model_dir)

    assert ok
    assert details["valid_qwen2p5_7b_shape"] is True
    assert details["weight_file_count"] == 1


def test_phase1_artifact_check_requires_each_clean_session(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    for relative in PHASE1_SESSIONS:
        session = artifact_root / "phase1" / relative
        for name in (
            "environment.json",
            "resolved_config.json",
            "session_summary.json",
            "status.json",
        ):
            path = session / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
        result = (
            session
            / "groups"
            / "qwen2p5_7b_decode_bs1_t1_bf16"
            / "result.json"
        )
        result.parent.mkdir(parents=True)
        result.write_text("{}\n")

    ok, details = _check_phase1_artifacts(artifact_root)

    assert ok
    assert len(details["sessions"]) == 3
