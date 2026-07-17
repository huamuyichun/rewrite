#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rewrite_selector.rewrites.registry import enumerate_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/rewrites/mlp_bounded_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    result = enumerate_from_config(config)
    result["config"] = config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "family_id",
                    "num_enumerated_states",
                    "num_valid_states",
                    "num_fx_unique_before_budget",
                    "num_fx_unique_retained",
                    "budget_truncated",
                    "candidate_growth_by_depth",
                )
                if key in result
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
