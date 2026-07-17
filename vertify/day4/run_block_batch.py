#!/usr/bin/env python3
"""Run Day 4 candidate instantiation for multiple block specs.

This wrapper exists to satisfy the Day 4 roadmap requirement: produce candidate
outputs for at least three blocks. It calls instantiate_candidates.py for each
block spec and aggregates candidate counts, dedup status, and equivalence status.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, default=Path(__file__).resolve().parent / "block_specs.json")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "block_outputs")
    parser.add_argument("--plans", type=Path, default=Path(__file__).resolve().parents[1] / "day3" / "candidate_plans.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    specs = json.loads(args.specs.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        block_id = spec["block_id"]
        block_out = args.out_dir / block_id
        cmd = [
            sys.executable,
            str(root / "instantiate_candidates.py"),
            "--plans",
            str(args.plans),
            "--out-dir",
            str(block_out),
            "--batch-size",
            str(spec["batch_size"]),
            "--seq-len",
            str(spec["seq_len"]),
            "--hidden-dim",
            str(spec["hidden_dim"]),
            "--intermediate-dim",
            str(spec["intermediate_dim"]),
            "--dtype",
            spec["dtype"],
            "--seed",
            str(spec["seed"]),
        ]
        subprocess.run(cmd, check=True)

        dedup = json.loads((block_out / "dedup_result.json").read_text(encoding="utf-8"))
        metadata = json.loads((block_out / "metadata.json").read_text(encoding="utf-8"))
        eq_rows = json.loads((block_out / "equivalence_results.json").read_text(encoding="utf-8"))
        candidate_rows = json.loads((block_out / "candidate_summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "block_id": block_id,
                "seq_len": spec["seq_len"],
                "hidden_dim": spec["hidden_dim"],
                "intermediate_dim": spec["intermediate_dim"],
                "dtype": spec["dtype"],
                "num_candidates": len(candidate_rows),
                "dedup_status": dedup["status"],
                "num_unique_actual_signatures": dedup["num_unique_actual_signatures"],
                "equivalence_status": metadata["equivalence_status"],
                "num_equivalence_passed": sum(1 for row in eq_rows if row["allclose_to_baseline"]),
                "out_dir": str(block_out),
            }
        )

    status = "ok"
    errors: list[str] = []
    if len(rows) < 3:
        status = "failed"
        errors.append("Day 4 requires at least 3 block candidate outputs")
    for row in rows:
        if row["num_candidates"] != 6:
            status = "failed"
            errors.append(f"{row['block_id']} has {row['num_candidates']} candidates, expected 6")
        if row["dedup_status"] != "ok":
            status = "failed"
            errors.append(f"{row['block_id']} dedup failed")
        if row["equivalence_status"] != "ok":
            status = "failed"
            errors.append(f"{row['block_id']} equivalence failed")

    result = {
        "status": status,
        "num_blocks": len(rows),
        "num_total_candidate_outputs": sum(int(row["num_candidates"]) for row in rows),
        "errors": errors,
        "blocks": rows,
    }
    write_csv(root / "block_candidate_summary.csv", rows)
    (root / "block_run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
