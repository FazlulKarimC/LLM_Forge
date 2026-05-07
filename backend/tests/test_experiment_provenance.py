"""Tests for shared experiment provenance helpers."""

from types import SimpleNamespace

from app.services.experiment_provenance import (
    build_effective_execution_manifest_entry,
    same_baseline_lineage,
)


def _experiment(**overrides):
    config = {
        "model_name": "mock-model",
        "dataset_name": "sample",
        "reasoning_method": "naive",
        "provider": "auto",
        "hyperparameters": {"temperature": 0.1, "max_tokens": 150, "seed": 42},
        "num_samples": 10,
        "routing": {"policy": "fallback_chain", "strict_comparison": True},
    }
    config.update(overrides.pop("config_updates", {}))
    return SimpleNamespace(
        dataset_name=overrides.pop("dataset_name", "sample"),
        model_name=overrides.pop("model_name", "mock-model"),
        dataset_hash=overrides.pop("dataset_hash", "dataset-hash"),
        config=config,
    )


def test_same_baseline_lineage_requires_execution_defining_fields():
    baseline = _experiment()
    candidate = _experiment(config_updates={"reasoning_method": "cot"})

    assert not same_baseline_lineage(baseline, candidate)


def test_effective_execution_manifest_entry_is_canonical_and_hashed():
    entry, entry_hash = build_effective_execution_manifest_entry(
        attempt=3,
        engine_type="auto (router)",
        provider="auto",
        routing_config={"policy": "fallback_chain", "strict_comparison": True},
        configured_hyperparameters={"temperature": 0.1, "max_tokens": 150, "seed": 7},
        effective_hyperparameters={"temperature": 0.1, "max_tokens": 512, "seed": 7},
        dataset_hash="dataset-hash",
        sample_ids=["sample-1"],
        sample_count=1,
        execution_mode="sequential",
        rag_enabled=False,
        optimization={"enable_batching": False},
    )

    assert entry["strict_comparison"] is True
    assert entry["adjustments"]["max_tokens"] == {
        "configured": 150,
        "effective": 512,
    }
    assert entry["sample_ids"] == ["sample-1"]
    assert entry_hash
