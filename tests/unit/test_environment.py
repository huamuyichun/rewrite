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


def test_manifest_records_portable_path_config(monkeypatch) -> None:
    monkeypatch.setattr(environment.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(environment, "_run", lambda command: "")
    monkeypatch.setenv("REWRITE_ROOT", "/workspace/rewrite")
    monkeypatch.setenv("REWRITE_ARTIFACT_ROOT", "/data/artifacts")
    monkeypatch.setenv("QWEN_MODEL_DIR", "/data/models/qwen")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    manifest = environment.environment_manifest()

    assert manifest["path_config"]["REWRITE_ROOT"] == "/workspace/rewrite"
    assert (
        manifest["path_config"]["REWRITE_ARTIFACT_ROOT"]
        == "/data/artifacts"
    )
    assert manifest["path_config"]["QWEN_MODEL_DIR"] == "/data/models/qwen"
    assert manifest["cuda_device_order"] == "PCI_BUS_ID"
    assert "triton" in manifest
    assert manifest["hardware_environment_domain"]["domain_id"].startswith("env_")
