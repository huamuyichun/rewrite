from pathlib import Path

from rewrite_selector.lowering.fingerprint import fingerprint_inductor_artifacts


def _write_trace(root: Path, source_name: str, session: str) -> None:
    root.mkdir()
    (root / "ir_pre_fusion.txt").write_text("op0 = mm(shape=[1, 16])\n")
    (root / "ir_post_fusion.txt").write_text("kernel0 = fused(op0)\n")
    (root / "fx_graph_transformed.py").write_text(f"# source: {source_name}\n")
    (root / "output_code.py").write_text(
        f"# kernel path: {root.resolve()}/{session}/kernel.py\n"
        "extern_kernels.mm(x, w, out=buf0)\n"
    )


def test_lowered_fingerprint_ignores_source_metadata(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_trace(first, "gate_up", "session_a")
    _write_trace(second, "packed", "session_b")
    first_result = fingerprint_inductor_artifacts(first)
    second_result = fingerprint_inductor_artifacts(second)
    assert first_result["fingerprint_schema_version"] == "inductor-ir-v3"
    assert first_result["lowered_sha256"] == second_result["lowered_sha256"]
    assert (
        first_result["generated_code_sha256"]
        == second_result["generated_code_sha256"]
    )
    assert first_result["execution_sha256"] == second_result["execution_sha256"]
