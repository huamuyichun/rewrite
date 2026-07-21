import pytest

from rewrite_selector.evaluation.execution_classes import (
    analyze_fingerprint_consistency,
    build_execution_classes,
    class_by_candidate,
    validate_complete_fingerprints,
)


def _audits():
    return {
        "p0": {"status": "ok", "lowered": {"execution_sha256": "a" * 64}},
        "p1": {"status": "ok", "lowered": {"execution_sha256": "a" * 64}},
        "p2": {"status": "ok", "lowered": {"execution_sha256": "b" * 64}},
        "invalid": {"status": "failed", "lowered": {"execution_sha256": "c" * 64}},
    }


def test_execution_class_mapping_and_canonical_representative() -> None:
    plans = [
        {"candidate_id": "p0", "rewrite_trace": []},
        {"candidate_id": "p1", "rewrite_trace": ["rule.z"]},
        {"candidate_id": "p2", "rewrite_trace": ["rule.a"]},
        {"candidate_id": "invalid", "rewrite_trace": []},
    ]
    classes = build_execution_classes(plans, _audits())
    assert len(classes) == 2
    duplicate = next(item for item in classes if len(item["candidate_ids"]) == 2)
    assert duplicate["canonical_candidate_id"] == "p0"
    assert duplicate["candidate_ids"] == ["p0", "p1"]
    mapping = class_by_candidate(classes)
    assert mapping["p0"] == mapping["p1"]
    assert mapping["p0"] != mapping["p2"]
    assert "invalid" not in mapping


def test_fingerprint_inconsistency_uses_noise_floor() -> None:
    plans = [
        {"candidate_id": "p0", "rewrite_trace": []},
        {"candidate_id": "p1", "rewrite_trace": ["rule.z"]},
        {"candidate_id": "p2", "rewrite_trace": ["rule.a"]},
        {"candidate_id": "invalid", "rewrite_trace": []},
    ]
    classes = build_execution_classes(plans, _audits())
    summary = {
        "p0": {"p50_ms": 1.0},
        "p1": {"p50_ms": 1.03},
    }
    result = analyze_fingerprint_consistency(classes, summary, 0.02)
    duplicate = next(
        item
        for item in result
        if len(item["candidate_p50_ms"]) == 2
    )
    assert duplicate["status"] == "fingerprint_inconsistency"


def test_missing_execution_fingerprint_cannot_form_singleton_class() -> None:
    plans = [{"candidate_id": "p0", "rewrite_trace": []}]
    audits = {"p0": {"status": "ok", "lowered": {}}}

    with pytest.raises(ValueError, match="missing fingerprints: p0"):
        build_execution_classes(plans, audits)


def test_complete_fingerprint_gate_reports_all_missing_fields() -> None:
    audits = {
        "p0": {
            "status": "ok",
            "lowered": {
                "fingerprint_schema_version": "inductor-ir-v3",
                "artifact_files": ["output_code.py"],
                "lowered_sha256": "lowered",
                "generated_code_sha256": "generated",
                "execution_sha256": "execution",
            },
        },
        "p1": {
            "status": "ok",
            "lowered": {
                "fingerprint_schema_version": "inductor-ir-v3",
                "artifact_files": [],
                "lowered_sha256": None,
                "generated_code_sha256": None,
                "execution_sha256": None,
            },
        },
    }

    with pytest.raises(ValueError, match="incomplete lowering fingerprints: p1"):
        validate_complete_fingerprints(audits, "inductor-ir-v3")
