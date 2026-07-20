from pathlib import Path

import scripts.run_phase1_audit as audit_runner
from scripts.run_phase1_audit import (
    configured_artifact_root,
    configured_registry_path,
    path_for_record,
    registry_entry,
)


def test_configured_paths_follow_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact_root = tmp_path / "external-artifacts"
    registry = tmp_path / "metadata" / "registry.jsonl"
    monkeypatch.setenv("REWRITE_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("REWRITE_REGISTRY_PATH", str(registry))

    assert configured_artifact_root() == artifact_root
    assert configured_registry_path() == registry


def test_registry_entry_supports_external_artifact_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    monkeypatch.setattr(audit_runner, "ROOT", repository_root)

    session_dir = tmp_path / "external-artifacts" / "run" / "session"
    session_dir.mkdir(parents=True)
    resolved_config = session_dir / "resolved_config.json"
    resolved_config.write_text("{}\n")

    entry = registry_entry(
        run_id="run",
        session_id="session",
        status="ok",
        source={"commit": "abc", "dirty": False},
        resolved_config=resolved_config,
        session_dir=session_dir,
        cache_policy="cold_session",
        cache_preexisting=False,
    )

    assert entry["artifact_path"] == str(session_dir.resolve())
    assert entry["environment_manifest_path"] == str(
        (session_dir / "environment.json").resolve()
    )
    assert path_for_record(repository_root / "artifacts") == "artifacts"
