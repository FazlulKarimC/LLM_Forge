"""Helpers for experiment comparability and execution provenance."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional, Sequence


BASELINE_LINEAGE_CONFIG_KEYS = (
    "reasoning_method",
    "provider",
    "hyperparameters",
    "rag",
    "agent",
    "optimization",
    "graders",
    "routing",
    "prompt_version_id",
    "num_samples",
)


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation for nested provenance comparisons."""
    return json.dumps(value, sort_keys=True, default=str)


def _mapping_or_dump(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return value


def baseline_lineage_key(experiment: Any) -> dict[str, Any]:
    """Return the fields that define a fair automatic regression baseline."""
    config = getattr(experiment, "config", None) or {}
    comparable_config = {
        key: config.get(key)
        for key in BASELINE_LINEAGE_CONFIG_KEYS
    }
    return {
        "dataset_name": getattr(experiment, "dataset_name", None) or config.get("dataset_name"),
        "model_name": getattr(experiment, "model_name", None) or config.get("model_name"),
        "dataset_hash": getattr(experiment, "dataset_hash", None),
        "config": comparable_config,
    }


def same_baseline_lineage(baseline: Any, candidate: Any) -> bool:
    """Whether two experiments are comparable for automatic regression gates."""
    return canonical_json(baseline_lineage_key(baseline)) == canonical_json(
        baseline_lineage_key(candidate)
    )


def build_effective_execution_manifest_entry(
    *,
    attempt: int,
    engine_type: str,
    provider: str,
    routing_config: Any,
    configured_hyperparameters: Mapping[str, Any],
    effective_hyperparameters: Mapping[str, Any],
    dataset_hash: Optional[str],
    sample_ids: Optional[Sequence[str]],
    sample_count: int,
    execution_mode: str,
    rag_enabled: bool,
    optimization: Any,
) -> tuple[dict[str, Any], str]:
    """Build a canonical effective-execution manifest entry and content hash."""
    routing_payload = _mapping_or_dump(routing_config)
    optimization_payload = _mapping_or_dump(optimization) or {}
    configured_hp = dict(configured_hyperparameters)
    effective_hp = dict(effective_hyperparameters)

    adjustments: dict[str, Any] = {}
    if configured_hp.get("max_tokens") != effective_hp.get("max_tokens"):
        adjustments["max_tokens"] = {
            "configured": configured_hp.get("max_tokens"),
            "effective": effective_hp.get("max_tokens"),
        }
    if configured_hp.get("temperature") != effective_hp.get("temperature"):
        adjustments["temperature"] = {
            "configured": configured_hp.get("temperature"),
            "effective": effective_hp.get("temperature"),
            "reason": "provider_minimum",
        }

    effective_execution = {
        "attempt": attempt,
        "engine_type": engine_type,
        "provider": provider,
        "routing": routing_payload,
        "strict_comparison": (
            (routing_payload or {}).get("strict_comparison", True)
            if provider == "auto"
            else None
        ),
        "hyperparameters": effective_hp,
        "configured_hyperparameters": configured_hp,
        "adjustments": adjustments,
        "dataset_hash": dataset_hash,
        "sample_ids": list(sample_ids or []),
        "sample_count": sample_count,
        "execution_mode": execution_mode,
        "rag_enabled": rag_enabled,
        "optimization": optimization_payload,
    }
    manifest_hash = hashlib.sha256(
        canonical_json(effective_execution).encode()
    ).hexdigest()
    return effective_execution, manifest_hash
