#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rewrite_selector.evaluation.phase1_analysis import analyze_sessions


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n")


def monitor_assessment(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "status": "missing",
            "pass": False,
            "reason": "no clean within-process self-effect result",
        }
    result = json.loads(path.read_text())
    contaminated = bool(result.get("contaminated", True))
    max_delta = float(result["max_absolute_relative_delta"])
    clock_stable = bool(result.get("clock_stable", False))
    passed = not contaminated and max_delta <= 0.005 and clock_stable
    return {
        "status": "pass" if passed else "failed",
        "pass": passed,
        "contaminated": contaminated,
        "median_relative_delta": result["median_relative_delta"],
        "median_absolute_relative_delta": result[
            "median_absolute_relative_delta"
        ],
        "max_absolute_relative_delta": max_delta,
        "clock_stable": clock_stable,
        "protocol": result["protocol"],
        "artifact_path": str(path),
        "reason": (
            None
            if passed
            else "contamination, clock variation, or >0.5% paired effect"
        ),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([head, rule, *body])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, action="append", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--monitor-result", type=Path)
    parser.add_argument(
        "--noise-floor-relative",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/reports/phase1",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cross = analyze_sessions(
        args.session,
        args.group_id,
        args.noise_floor_relative,
    )
    monitor = monitor_assessment(args.monitor_result)

    measurement = {
        "schema_version": "measurement-noise-report-v1",
        "group_id": args.group_id,
        "session_ids": cross["session_ids"],
        "num_sessions": cross["num_sessions"],
        "noise_floor_relative": cross["noise_floor_relative"],
        "class_table": cross["class_table"],
        "candidate_table": cross["candidate_table"],
        "pairwise": cross["pairwise"],
        "strict_pair_order_reproducibility": cross[
            "strict_pair_order_reproducibility"
        ],
        "strict_pair_count": cross["strict_pair_count"],
        "tie_pair_count": cross["tie_pair_count"],
        "ambiguous_pair_count": cross["ambiguous_pair_count"],
        "contaminated_round_ratio": cross[
            "contaminated_round_ratio"
        ],
        "all_sessions_clean": cross["all_sessions_clean"],
        "class_session_drift": cross["class_session_drift"],
        "monitor_self_effect": monitor,
    }
    lowering = {
        "schema_version": "lowering-collapse-report-v1",
        "group_id": args.group_id,
        "session_ids": cross["session_ids"],
        "execution_classes": cross["execution_classes"],
        "fingerprint_stability": cross["fingerprint_stability"],
        "all_fingerprints_stable": cross[
            "all_fingerprints_stable"
        ],
        "execution_class_mapping_stable": cross[
            "execution_class_mapping_stable"
        ],
        "execution_class_winner_counts": cross[
            "execution_class_winner_counts"
        ],
        "execution_class_winner_reproducibility": cross[
            "execution_class_winner_reproducibility"
        ],
        "p3_p4_p5": cross["p3_p4_p5"],
    }
    fixed = {
        "schema_version": "fixed-baseline-report-v1",
        "group_id": args.group_id,
        "session_ids": cross["session_ids"],
        "fixed_execution_class_rows": cross["fixed_class_rows"],
        "best_fixed_execution_class": cross[
            "best_fixed_execution_class"
        ],
        "baseline_to_best_gains": cross[
            "baseline_to_best_gains"
        ],
        "baseline_to_best_gain_median": cross[
            "baseline_to_best_gain_median"
        ],
        "baseline_to_best_gain_min": cross[
            "baseline_to_best_gain_min"
        ],
    }

    p3_tie = bool(
        cross["p3_p4_p5"].get("p3_p4_collapsed")
        and cross["p3_p4_p5"].get("p3_p5_label") == "tie"
    )
    answers = {
        "profiler_reproducible": (
            cross["num_sessions"] >= 3
            and cross["all_sessions_clean"]
            and cross["strict_pair_order_reproducibility"] >= 0.9
        ),
        "monitor_non_interfering": monitor["pass"],
        "fingerprints_stable": (
            cross["all_fingerprints_stable"]
            and cross["execution_class_mapping_stable"]
        ),
        "strict_pairs_reproducible": (
            cross["strict_pair_order_reproducibility"] >= 0.9
        ),
        "p3_p4_p5_stable_tie": p3_tie,
        "qwen_baseline_to_best_gain_exists": (
            cross["num_sessions"] >= 3
            and cross["baseline_to_best_gain_min"]
            > cross["noise_floor_relative"]
        ),
        "four_execution_classes_have_stable_ordering": (
            len(cross["execution_classes"]) == 4
            and cross["strict_pair_count"] > 0
            and cross["strict_pair_order_reproducibility"] >= 0.9
        ),
    }
    allow_discovery = all(answers.values())
    exit_decision = {
        "schema_version": "phase1-exit-decision-v1",
        "group_id": args.group_id,
        "session_ids": cross["session_ids"],
        "answers": answers,
        "allow_candidate_family_discovery": allow_discovery,
        "decision": (
            "pass_phase1"
            if allow_discovery
            else "hold_phase1_and_fix_failed_gates"
        ),
        "failed_gates": [
            key for key, value in answers.items() if not value
        ],
    }

    write_json(args.output_dir / "measurement_noise_report.json", measurement)
    write_json(args.output_dir / "lowering_collapse_report.json", lowering)
    write_json(args.output_dir / "fixed_baseline_report.json", fixed)
    write_json(args.output_dir / "phase1_exit_decision.json", exit_decision)

    class_rows = [
        [
            row["session_id"],
            row["execution_class_id"],
            f'{float(row["p50_ms"]):.6f}',
            f'{float(row["median_ci95_low_ms"]):.6f}',
            f'{float(row["median_ci95_high_ms"]):.6f}',
            f'{float(row["cv"]):.4f}',
        ]
        for row in measurement["class_table"]
    ]
    write_text(
        args.output_dir / "measurement_noise_report.md",
        "# Measurement Noise Report\n\n"
        f"Sessions: {', '.join(cross['session_ids'])}\n\n"
        f"Locked relative noise floor: "
        f"{100 * cross['noise_floor_relative']:.2f}%\n\n"
        + markdown_table(
            ["session", "execution class", "p50 ms", "CI low", "CI high", "CV"],
            class_rows,
        )
        + "\n\n"
        f"Strict/tie/ambiguous pairs: {cross['strict_pair_count']}/"
        f"{cross['tie_pair_count']}/{cross['ambiguous_pair_count']}.\n\n"
        f"Strict-pair order reproducibility: "
        f"{cross['strict_pair_order_reproducibility']:.3f}.\n\n"
        f"Contaminated round ratio: "
        f"{cross['contaminated_round_ratio']:.3f}.\n\n"
        f"Monitor self-effect status: {monitor['status']}.",
    )
    collapse_rows = [
        [
            item["execution_class_id"],
            item["canonical_candidate_id"],
            ", ".join(item["candidate_ids"]),
        ]
        for item in lowering["execution_classes"]
    ]
    write_text(
        args.output_dir / "lowering_collapse_report.md",
        "# Lowering Collapse Report\n\n"
        + markdown_table(
            ["execution class", "canonical representative", "candidates"],
            collapse_rows,
        )
        + "\n\n"
        f"All fingerprints stable: "
        f"{lowering['all_fingerprints_stable']}.\n\n"
        f"Execution-class mapping stable: "
        f"{lowering['execution_class_mapping_stable']}.\n\n"
        f"p3/p4 collapsed: "
        f"{lowering['p3_p4_p5'].get('p3_p4_collapsed')}; "
        f"p3/p5 label: "
        f"{lowering['p3_p4_p5'].get('p3_p5_label')}.",
    )
    fixed_rows = [
        [
            row["execution_class_id"],
            f'{100 * float(row["median_regret"]):.3f}%',
            f'{100 * float(row["p90_regret"]):.3f}%',
            f'{100 * float(row["max_regret"]):.3f}%',
            f'{float(row["win_share"]):.3f}',
        ]
        for row in fixed["fixed_execution_class_rows"]
    ]
    write_text(
        args.output_dir / "fixed_baseline_report.md",
        "# Fixed Baseline Report\n\n"
        + markdown_table(
            ["execution class", "median regret", "P90 regret", "max regret", "win share"],
            fixed_rows,
        )
        + "\n\n"
        f"Median baseline-to-best gain: "
        f"{100 * fixed['baseline_to_best_gain_median']:.3f}%.\n\n"
        f"Minimum session gain: "
        f"{100 * fixed['baseline_to_best_gain_min']:.3f}%.",
    )
    answer_rows = [
        [key, value] for key, value in answers.items()
    ]
    write_text(
        args.output_dir / "phase1_exit_decision.md",
        "# Phase 1 Exit Decision\n\n"
        + markdown_table(["gate", "result"], answer_rows)
        + "\n\n"
        f"Decision: **{exit_decision['decision']}**.\n\n"
        f"Candidate-family discovery allowed: "
        f"**{allow_discovery}**.",
    )
    print(json.dumps(exit_decision, indent=2))


if __name__ == "__main__":
    main()
