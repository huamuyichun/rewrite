#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rewrite_selector.evaluation.phase2_analysis import analyze_registry_sessions


def _percent(value: float) -> str:
    return f"{100 * float(value):.3f}%"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def _plan_label(analysis: dict[str, Any], plan_id: str | None) -> str:
    if plan_id is None:
        return "none"
    plan = analysis["semantic_plan_definitions"][plan_id]
    return f'{plan["label"]} (`{plan_id}`)'


def render_markdown(analysis: dict[str, Any]) -> str:
    all_scope = analysis["scopes"]["all"]
    questions = analysis["questions"]
    domain = analysis["hardware_environment_domain"]
    group_rows = []
    for group in analysis["groups"]:
        counts = group["counts"]
        pairs = group["pair_counts"]
        group_rows.append(
            [
                f'`{group["group_id"]}`',
                ",".join(group["session_ids"]),
                f'{counts["enumerated"]}/{counts["valid"]}/'
                f'{counts["fx_unique"]}/{counts["lowered_unique"]}/'
                f'{counts["execution_unique"]}',
                f'{pairs["strict"]}/{pairs["tie"]}/{pairs["ambiguous"]}',
                group["effective_competing_execution_classes"],
                _percent(group["best_worst_spread"]),
                _percent(group["baseline_to_point_oracle_gain"]),
                _percent(group["production_to_noise_aware_oracle_gain"]),
                "yes" if group["fingerprint_stable"] else "no",
            ]
        )
    plan_rows = []
    for row in all_scope["semantic_plan_rows"]:
        plan = analysis["semantic_plan_definitions"][row["semantic_plan_id"]]
        raw_regret = row["raw_regret"]
        noise_regret = row["noise_aware_regret"]
        plan_rows.append(
            [
                f'`{row["semantic_plan_id"]}`',
                plan["label"],
                _percent(row["strict_win_share"]),
                _percent(row["possible_win_share"]),
                _percent(row["fractional_tie_aware_win_share"]),
                _percent(raw_regret["p50"]),
                _percent(raw_regret["p90"]),
                _percent(raw_regret["max"]),
                _percent(noise_regret["max"]),
            ]
        )
    scope_rows = []
    for scope_name in ("all", "decode", "prefill"):
        scope = analysis["scopes"][scope_name]
        fixed = scope["best_fixed_semantic_plan"]
        raw_regret = fixed["raw_regret"]
        noise_regret = fixed["noise_aware_regret"]
        scope_rows.append(
            [
                scope_name,
                _plan_label(analysis, fixed["semantic_plan_id"]),
                _percent(raw_regret["p50"]),
                _percent(raw_regret["p90"]),
                _percent(raw_regret["max"]),
                _percent(noise_regret["max"]),
                _percent(fixed["possible_win_share"]),
            ]
        )
    top_k_rows = []
    for row in all_scope["top_k_oracle_curves"]:
        raw_regret = row["raw_regret"]
        noise_regret = row["noise_aware_regret"]
        top_k_rows.append(
            [
                row["k"],
                ", ".join(_plan_label(analysis, plan_id) for plan_id in row["semantic_plan_ids"]),
                _percent(raw_regret["p50"]),
                _percent(raw_regret["p90"]),
                _percent(raw_regret["max"]),
                _percent(noise_regret["max"]),
            ]
        )
    rule_rows = []
    for row in analysis["simple_rule_diagnostics"]:
        raw_regret = row["raw_regret"]
        noise_regret = row["noise_aware_regret"]
        rule_rows.append(
            [
                row["rule"],
                row["num_buckets"],
                _percent(raw_regret["p50"]),
                _percent(raw_regret["p90"]),
                _percent(raw_regret["max"]),
                _percent(noise_regret["max"]),
            ]
        )
    provenance_rows = [
        [
            f'`{row["session_id"]}`',
            f'`{row["source_commit"][:12]}`',
            f'`{row["config_sha256"][:12]}`',
            row["gpu_uuid"],
            row["triton"]["value"],
            row["triton"]["binding_source"],
            "pass" if row["provenance_complete"] else "fail",
        ]
        for row in analysis["session_audits"]
    ]
    replication_rows = []
    for group in analysis["groups"]:
        if group["num_sessions"] < 2:
            continue
        reproducibility = group["session_reproducibility"]
        gain = reproducibility["baseline_to_point_oracle_gain"]
        drift = reproducibility["session_drift"]
        replication_rows.append(
            [
                f'`{group["group_id"]}`',
                group["num_sessions"],
                _percent(reproducibility["point_winner_reproducibility"]),
                _percent(reproducibility["best_set_exact_reproducibility"]),
                _percent(
                    reproducibility["pairwise_point_order_reproducibility"]
                ),
                f'{_percent(gain["median"])} '
                f'[{_percent(gain["min"])}, {_percent(gain["max"])}]',
                _percent(drift["max_class_p50_range"]),
                "yes" if group["execution_class_mapping_stable"] else "no",
                "yes" if group["fingerprint_stable"] else "no",
                _percent(group["contaminated_session_ratio"]),
            ]
        )
    production_raw_gain = all_scope["production_to_point_oracle_gain"]
    production_noise_gain = all_scope["production_to_noise_aware_oracle_gain"]
    fixed = all_scope["best_fixed_semantic_plan"]
    fixed_raw_regret = fixed["raw_regret"]
    fixed_noise_regret = fixed["noise_aware_regret"]
    strict_counts = questions["strictly_distinguishable_class_counts"]
    lines = [
        "# RMSNorm Discovery 离线聚合",
        "",
        "## 结论",
        "",
        f'- 分析域：`{domain["domain_id"]}`；迁移 tag：`{domain["migration_tag"]}`。',
        f'- 8 FX → 6 execution 的 retention 在全部 {len(analysis["groups"])} 个 group 中'
        f'{"稳定" if questions["retention_stable_8_to_6"] else "不稳定"}。',
        f'- best-worst spread 超过锁定 2% floor 的 group：'
        f'{questions["groups_with_spread_over_noise_floor"]}/{len(analysis["groups"])}；'
        f'存在至少一条 strict pair 的 group：'
        f'{questions["groups_with_strict_preference"]}/{len(analysis["groups"])}。',
        f'- 每组参与 strict pair 的 execution class 数：{strict_counts}；noise-aware best set '
        f'平均包含 {questions["average_noise_aware_best_set_size"]:.3f} 个 execution class。',
        f'- noise-aware best semantic-plan set '
        f'{"随 group 变化" if questions["winner_semantic_plan_sets_vary"] else "不随 group 变化"}，'
        f'但所有 group 共同保留 {len(questions["always_possible_semantic_plan_ids"])} 个 possible plans，'
        f'唯一 strict semantic winner 数为 {len(questions["strict_winner_semantic_plan_ids"])}。',
        f'- global best fixed：{_plan_label(analysis, fixed["semantic_plan_id"])}；possible-win share '
        f'{_percent(fixed["possible_win_share"])}，raw P50/P90/max regret 为 '
        f'{_percent(fixed_raw_regret["p50"])}/{_percent(fixed_raw_regret["p90"])}/'
        f'{_percent(fixed_raw_regret["max"])}，noise-aware max 为 '
        f'{_percent(fixed_noise_regret["max"])}。',
        f'- production/default 到 point oracle 的 median/P90/max gain 为 '
        f'{_percent(production_raw_gain["median"])}/{_percent(production_raw_gain["p90"])}/'
        f'{_percent(production_raw_gain["max"])}；到 noise-aware oracle 为 '
        f'{_percent(production_noise_gain["median"])}/{_percent(production_noise_gain["p90"])}/'
        f'{_percent(production_noise_gain["max"])}。',
        f'- 初步 context-sensitive selection 证据：'
        f'{"存在，但必须选择性复测" if questions["preliminary_context_sensitive_value"] else "不足；fixed/simple policy 已接近 oracle，复测用于确认降级结论"}。',
        "",
        "所有绝对 latency 只在该 hardware/environment domain 内聚合；旧服务器仅可比较"
        " normalized gain、排序与 fingerprint。锁定的 2% noise floor 未修改。",
        "",
        "## 统计口径",
        "",
        "`semantic_plan_id` 由 family、canonical rewrite trace 与语义参数的 canonical JSON"
        " 哈希生成。它跨 workload 稳定，再按 group 多对一映射到 execution class。pairwise"
        " 相对差异按 blocked round 配对 bootstrap；95% CI 整体越过 ±2% 才是 strict，"
        "完全落在区间内是 tie，其余是 ambiguous。noise-aware regret 对 best set 内计划记 0。",
        "",
        "## Group 结果",
        "",
        _table(
            [
                "group",
                "sessions",
                "enum/valid/FX/lowered/exec",
                "strict/tie/ambiguous",
                "best-set classes",
                "spread",
                "default→point",
                "default→noise-aware",
                "fingerprint stable",
            ],
            group_rows,
        ),
        "",
        "每个 execution class 的 candidates、semantic plans、raw sample count、P50/mean/CV、"
        "bootstrap CI 和所有 relative-difference CI 保存在对应 JSON；CSV 为一行一个"
        " group-local execution class。",
        "",
        "## Semantic Plans",
        "",
        _table(
            [
                "semantic plan",
                "label",
                "strict win",
                "possible win",
                "fractional win",
                "raw P50",
                "raw P90",
                "raw max",
                "noise max",
            ],
            plan_rows,
        ),
        "",
        "## Fixed Baselines",
        "",
        _table(
            [
                "scope",
                "best fixed",
                "raw P50",
                "raw P90",
                "raw max",
                "noise max",
                "possible win",
            ],
            scope_rows,
        ),
        "",
        f'fractional winner entropy：{all_scope["winner_entropy"]:.6f}（normalized '
        f'{all_scope["winner_entropy_normalized"]:.6f}）。tie group 没有被强制指定唯一 winner。',
        "",
        "## Top-k Oracle",
        "",
        _table(
            ["k", "semantic plan portfolio", "raw P50", "raw P90", "raw max", "noise max"],
            top_k_rows,
        ),
        "",
        "## 简单规则诊断",
        "",
        _table(
            ["rule", "buckets", "raw P50", "raw P90", "raw max", "noise max"],
            rule_rows,
        ),
        "",
        "这些规则只在当前 8 个 group 上做同集 diagnostic，不是训练结果，也不代表"
        " held-out 泛化。exact shape bucket 仍让两个 context 共用一个固定计划，避免一组"
        "一个规则的无意义拟合。",
        "",
        "## Provenance",
        "",
        _table(
            ["session", "commit", "config", "GPU UUID", "Triton", "binding", "audit"],
            provenance_rows,
        ),
        "",
        "历史 discovery manifest 未直接写入 Triton 和 CUDA device order；二者由冻结的"
        " environment-domain record 与迁移报告补充绑定。新 runner 已直接记录这些字段。",
        "",
        "## 跨 Session 复现性",
        "",
        (
            _table(
                [
                    "group",
                    "sessions",
                    "point winner",
                    "best-set exact",
                    "pair order",
                    "default→point median [min,max]",
                    "max class drift",
                    "mapping stable",
                    "fingerprint stable",
                    "contaminated",
                ],
                replication_rows,
            )
            if replication_rows
            else "尚无 group 拥有两个或以上独立 session。"
        ),
        "",
        "point winner、best-set exact 和 pair order 分别报告点赢家、noise-aware best set"
        " 完全一致率，以及 15 个 execution-class pair 的 P50 顺序复现率。class drift 是"
        " 同一 fingerprint class 的跨 session P50 最大相对范围。详细的 per-session"
        " best set、pairwise order、gain 和 drift 保存在 JSON。",
        "",
        "## 限制",
        "",
        (
            "当前每组只有一个完整独立 session。CI 反映 blocked-round measurement uncertainty，"
            "尚不能证明跨 session 可复现性；family 保留/降级/淘汰决定必须等待 adaptive"
            " replication 完成。"
            if not replication_rows
            else "adaptive groups 的跨 session CI 以 session 为外层 bootstrap 单位；仍只有一个"
            " session 的非复测 groups 只作为单 session tie/ambiguous evidence。主要结论需等"
            "预注册 groups 补足三个独立 session。"
        ),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_csv(path: Path, analysis: dict[str, Any]) -> None:
    fieldnames = [
        "group_id",
        "phase",
        "context",
        "shape_bucket",
        "session_ids",
        "execution_class_id",
        "class_signature",
        "candidate_ids",
        "semantic_plan_ids",
        "raw_sample_count",
        "p50_ms",
        "mean_ms",
        "cv",
        "ci95_low_ms",
        "ci95_high_ms",
        "noise_aware_best",
        "production_default_class",
        "fingerprint_stable",
        "contaminated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in analysis["groups"]:
            for row in group["execution_classes"]:
                writer.writerow(
                    {
                        "group_id": group["group_id"],
                        "phase": group["workload"]["phase"],
                        "context": group["workload"].get("context"),
                        "shape_bucket": group["shape_bucket"],
                        "session_ids": ";".join(group["session_ids"]),
                        "execution_class_id": row["execution_class_id"],
                        "class_signature": row["class_signature"],
                        "candidate_ids": ";".join(row["candidate_ids"]),
                        "semantic_plan_ids": ";".join(row["semantic_plan_ids"]),
                        "raw_sample_count": row["raw_sample_count"],
                        "p50_ms": row["p50_ms"],
                        "mean_ms": row["mean_ms"],
                        "cv": row["cv"],
                        "ci95_low_ms": row["median_ci95_low_ms"],
                        "ci95_high_ms": row["median_ci95_high_ms"],
                        "noise_aware_best": row["class_signature"]
                        in group["noise_aware_best_class_signatures"],
                        "production_default_class": row["class_signature"]
                        == group["baseline_class_signature"],
                        "fingerprint_stable": row["fingerprint_stable"],
                        "contaminated": group["contaminated"],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 2 discovery sessions by semantic plan and execution class"
    )
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "artifacts" / "registry.jsonl",
    )
    parser.add_argument(
        "--environment-domain",
        type=Path,
        default=ROOT / "configs" / "environments" / "target_a100_20260720.json",
    )
    parser.add_argument("--noise-floor-relative", type=float, default=0.02)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "reports" / "phase2",
    )
    parser.add_argument("--basename", default="rmsnorm_discovery_summary")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis = analyze_registry_sessions(
        ROOT,
        args.registry,
        args.environment_domain,
        set(args.run_id),
        noise_floor_relative=args.noise_floor_relative,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    json_path = args.output_dir / f"{args.basename}.json"
    csv_path = args.output_dir / f"{args.basename}.csv"
    markdown_path = args.output_dir / f"{args.basename}.md"
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    write_csv(csv_path, analysis)
    markdown_path.write_text(render_markdown(analysis))
    print(
        json.dumps(
            {
                "status": "ok",
                "run_ids": sorted(set(args.run_id)),
                "groups": len(analysis["groups"]),
                "sessions": len(analysis["session_audits"]),
                "all_session_provenance_complete": analysis[
                    "all_session_provenance_complete"
                ],
                "outputs": [str(markdown_path), str(json_path), str(csv_path)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
