from rewrite_selector.evaluation.execution_classes import (
    analyze_fingerprint_consistency,
    build_execution_classes,
    class_by_candidate,
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
