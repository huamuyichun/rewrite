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

from rewrite_selector.evaluation.family_decision import build_rmsnorm_decision


def _percent(value: float) -> str:
    return f"{100 * float(value):.3f}%"


def _plan_label(analysis: dict[str, Any], plan_id: str) -> str:
    plan = analysis["semantic_plan_definitions"][plan_id]
    return f'{plan["label"]} (`{plan_id}`)'


def render_markdown(decision: dict[str, Any], analysis: dict[str, Any]) -> str:
    evidence = decision["evidence"]
    fixed = evidence["global_best_fixed"]
    decode = evidence["decode_best_fixed"]
    prefill = evidence["prefill_best_fixed"]
    production_point = evidence["production_to_point_oracle_gain"]
    production_noise = evidence["production_to_noise_aware_oracle_gain"]
    simple = evidence["best_simple_rule"]
    replication_rows = []
    for row in decision["replication"]:
        gain = row["baseline_to_point_oracle_gain"]
        drift = row["session_drift"]
        replication_rows.append(
            "| `{}` | {} | {} | {} | {} | {} [{}, {}] | {} | {} |".format(
                row["group_id"],
                len(row["session_ids"]),
                _percent(row["point_winner_reproducibility"]),
                _percent(row["best_set_exact_reproducibility"]),
                _percent(row["pairwise_point_order_reproducibility"]),
                _percent(gain["median"]),
                _percent(gain["min"]),
                _percent(gain["max"]),
                _percent(drift["max_class_p50_range"]),
                "yes"
                if row["fingerprint_stable"]
                and row["execution_class_mapping_stable"]
                else "no",
            )
        )
    return "\n".join(
        [
            "# RMSNorm Family 正式决策",
            "",
            "## 决策",
            "",
            f'**{decision["decision"]["code"]}. {decision["decision"]["label"]}。**',
            "",
            "RMSNorm 保留为 equivalence、lowering-collapse、fingerprint 和 measurement"
            " diagnostic family；不作为 learned context-sensitive selector 的主要训练空间，"
            "不再通过扩相似 shape 或 rewrite 数量规避该结论。",
            "",
            "## 证据",
            "",
            f'- 8/8 groups 均为 8 FX → 6 execution，retention '
            f'{_percent(evidence["execution_retention"]["median"])}；20 个正式 session '
            "全部通过 provenance，新增 12 个 session 无 aborted/contaminated。",
            f'- aggregate pair 为 {evidence["pair_counts"]["strict"]} strict / '
            f'{evidence["pair_counts"]["tie"]} tie / '
            f'{evidence["pair_counts"]["ambiguous"]} ambiguous；每组有 5-6 个 class '
            "参与 strict pair，execution diversity 和性能差异真实存在。",
            f'- 唯一 strict semantic winner 数为 '
            f'{len(evidence["strict_winner_semantic_plan_ids"])}；五个核心 semantic plans '
            "在 8/8 groups 都是 possible winner。point winner 的变化没有形成跨 context"
            " 可复现的 strict semantic winner exchange。",
            f'- global fixed 为 '
            f'{_plan_label(analysis, fixed["semantic_plan_id"])}，raw P50/P90/max regret '
            f'{_percent(fixed["raw_regret"]["p50"])}/'
            f'{_percent(fixed["raw_regret"]["p90"])}/'
            f'{_percent(fixed["raw_regret"]["max"])}，noise-aware max '
            f'{_percent(fixed["noise_aware_regret"]["max"])}。',
            f'- decode fixed 为 {_plan_label(analysis, decode["semantic_plan_id"])}，'
            f'raw max regret {_percent(decode["raw_regret"]["max"])}；prefill fixed 为 '
            f'{_plan_label(analysis, prefill["semantic_plan_id"])}，raw max regret '
            f'{_percent(prefill["raw_regret"]["max"])}。',
            f'- 最简单近似 oracle 规则 `{simple["rule"]}` 的 raw P90/max regret 为 '
            f'{_percent(simple["raw_regret"]["p90"])}/'
            f'{_percent(simple["raw_regret"]["max"])}。',
            f'- production/default 到 point oracle 的跨 group median/P90/max gain 为 '
            f'{_percent(production_point["median"])}/'
            f'{_percent(production_point["p90"])}/'
            f'{_percent(production_point["max"])}；到 noise-aware oracle 为 '
            f'{_percent(production_noise["median"])}/'
            f'{_percent(production_noise["p90"])}/'
            f'{_percent(production_noise["max"])}。production 在每组 noise-aware best set 中。',
            f'- same-class 短 diagnostic warning 共 '
            f'{evidence["same_class_timing_warning_count"]} 条；fingerprint 和 semantic→execution '
            "mapping 跨 session 全部稳定，因此它们作为 measurement variability 记录，"
            "不解释为 context-sensitive codegen。",
            "",
            "## 选择性复测",
            "",
            "| group | sessions | point winner | best-set exact | pair order | default→point median [min,max] | max class drift | mapping/fingerprint |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            *replication_rows,
            "",
            "`rms_p02` 的 point gain 为 0%/3.947%/0.667%，pair-order reproducibility 只有"
            " 60%，第三次没有复现第二次的大 gap；其 fingerprint、mapping、clock 和污染门禁"
            "均正常。这是 session variability 证据，不是稳定 winner exchange。",
            "",
            "## MLP 后续",
            "",
            "RMSNorm decision 已完成，允许开始 9 个 MLP control groups。每组先运行一个"
            "独立 screening session，先做 equivalence 和 FX/lowered/execution dedup，只正式"
            " profile execution-unique candidates。screening 聚合后才选择信息量高的 groups "
            "做 adaptive replication；不机械重复 19 个 FX candidates，不训练 selector。",
            "",
            "Phase 2 exit gate 仍为 `pending_mlp_control_discovery`。",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the RMSNorm family decision record")
    parser.add_argument(
        "--analysis",
        type=Path,
        default=ROOT / "docs/reports/phase2/rmsnorm_discovery_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/reports/phase2",
    )
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    decision = build_rmsnorm_decision(analysis, str(args.analysis.relative_to(ROOT)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "rmsnorm_family_decision.json"
    markdown_path = args.output_dir / "rmsnorm_family_decision.md"
    json_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(decision, analysis))
    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "outputs": [str(markdown_path), str(json_path)],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
