from rewrite_selector.profiling.blocked import blocked_schedule


def test_blocked_schedule_is_deterministic_and_complete() -> None:
    candidates = ["a", "b", "c", "d"]
    first = blocked_schedule(candidates, rounds=5, seed=11)
    second = blocked_schedule(candidates, rounds=5, seed=11)
    assert first == second
    assert all(sorted(round_order) == candidates for round_order in first)
    assert len({tuple(round_order) for round_order in first}) > 1

