from rewrite_selector.profiling import environment


def test_cpu_snapshot_has_stable_schema(monkeypatch) -> None:
    monkeypatch.setattr(environment.torch.cuda, "is_available", lambda: False)
    snapshot = environment.gpu_snapshot()
    assert snapshot["cuda_available"] is False
    assert snapshot["backend"] == "none"
    assert snapshot["foreign_processes"] == []


def test_is_contaminated() -> None:
    assert not environment.is_contaminated(
        [{"foreign_processes": []}, {"foreign_processes": []}]
    )
    assert environment.is_contaminated(
        [{"foreign_processes": [{"pid": 7}]}]
    )
