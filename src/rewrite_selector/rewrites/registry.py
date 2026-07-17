from __future__ import annotations

from typing import Any, Callable

from rewrite_selector.rewrites.mlp_enumerator import enumerate_mlp_candidates
from rewrite_selector.rewrites.rmsnorm_enumerator import (
    enumerate_rmsnorm_candidates,
)


ENUMERATORS: dict[str, Callable[..., dict[str, Any]]] = {
    "mlp_bounded": enumerate_mlp_candidates,
    "rmsnorm_bounded": enumerate_rmsnorm_candidates,
}


def enumerate_from_config(config: dict[str, Any]) -> dict[str, Any]:
    enumerator_name = str(config.get("enumerator"))
    try:
        enumerator = ENUMERATORS[enumerator_name]
    except KeyError as exc:
        raise ValueError(f"unsupported enumerator: {enumerator_name}") from exc
    result = enumerator(
        max_depth=int(config["max_rewrite_depth"]),
        max_candidates=int(config["max_fx_unique_candidates"]),
    )
    configured_family = str(config.get("family_id"))
    if result["family_id"] != configured_family:
        raise ValueError(
            "enumerator family mismatch: "
            f"config={configured_family}, result={result['family_id']}"
        )
    return result


def resolve_rewrite_config(config: dict[str, Any]) -> dict[str, Any]:
    if "plans" in config:
        return config
    enumeration = enumerate_from_config(config)
    return {
        **config,
        "plans": enumeration["candidates"],
        "enumeration_summary": {
            key: value
            for key, value in enumeration.items()
            if key not in {"candidates", "enumeration_tree"}
        },
    }
