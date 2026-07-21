from pathlib import Path

import torch

import scripts.run_phase1_audit as audit_runner


def test_compile_candidate_disables_compiler_caches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_states: list[bool] = []

    def fake_compile(module, *, backend):
        assert backend == "inductor"
        cache_states.append(torch.compiler.config.force_disable_caches)

        def compiled(value):
            cache_states.append(torch.compiler.config.force_disable_caches)
            return module(value)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        audit_runner,
        "fingerprint_inductor_artifacts",
        lambda _path: {"execution_sha256": "fingerprint"},
    )
    module = torch.nn.Identity()
    example = torch.ones(1)

    compiled, _, lowered = audit_runner.compile_candidate(
        module,
        example,
        tmp_path / "trace",
    )

    assert callable(compiled)
    assert lowered["execution_sha256"] == "fingerprint"
    assert cache_states == [True, True]
    assert torch.compiler.config.force_disable_caches is False
