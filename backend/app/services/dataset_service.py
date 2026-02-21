"""
Dataset Service

Abstracts dataset loading, sampling, and caching.
Supports multiple evaluation datasets for different reasoning methods.

Datasets:
- trivia_qa: Single-hop factual recall (100 questions)
- knowledge_base: Questions grounded in indexed articles (50 questions, RAG-focused)
- multi_hop: Multi-step reasoning questions (40 questions, CoT/ReAct-focused)
- math_reasoning: GSM8K-style word problems (40 questions, CoT/calculator)
- commonsense_qa: Everyday reasoning (30 questions)
- react_bench: Tool-requiring questions (30 questions, ReAct agent)
- sample: Built-in smoke test (10 questions)
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Optional, TypedDict

logger = logging.getLogger(__name__)


class DatasetExample(TypedDict):
    """Single dataset example."""
    id: str
    question: str
    answer: str
    aliases: List[str]


# =============================================================================
# Dataset Registry — metadata for all known datasets
# =============================================================================

DATASET_REGISTRY = {
    "trivia_qa": {
        "name": "TriviaQA",
        "description": "Single-hop factual recall questions",
        "category": "general",
        "recommended_for": ["naive", "cot"],
        "file": "triviaqa/trivia_qa.json",
    },
    "knowledge_base": {
        "name": "Knowledge Base QA",
        "description": "Questions answerable from indexed articles — ideal for RAG evaluation",
        "category": "rag",
        "recommended_for": ["naive", "cot", "rag"],
        "file": "knowledge_base/knowledge_base_qa.json",
    },
    "multi_hop": {
        "name": "Multi-Hop QA",
        "description": "Requires combining 2+ facts for multi-step reasoning",
        "category": "reasoning",
        "recommended_for": ["cot", "react"],
        "file": "multi_hop/multi_hop_qa.json",
    },
    "math_reasoning": {
        "name": "Math Reasoning",
        "description": "GSM8K-style word problems requiring arithmetic",
        "category": "reasoning",
        "recommended_for": ["cot", "react"],
        "file": "math_reasoning/math_reasoning.json",
    },
    "commonsense_qa": {
        "name": "Commonsense QA",
        "description": "Everyday knowledge and reasoning questions",
        "category": "general",
        "recommended_for": ["naive", "cot"],
        "file": "commonsense_qa/commonsense_qa.json",
    },
    "react_bench": {
        "name": "ReAct Agent Bench",
        "description": "Multi-tool questions requiring search + calculation",
        "category": "agent",
        "recommended_for": ["react"],
        "file": "react_bench/react_bench.json",
    },
}


class DatasetService:
    """
    Service for loading evaluation datasets.
    
    Supports multiple dataset types organized by category:
    - General: trivia_qa, commonsense_qa
    - RAG: knowledge_base
    - Reasoning: multi_hop, math_reasoning
    - Agent: react_bench
    """
    
    # Base paths
    _PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # LlmForge/
    _DATASETS_DIR = _PROJECT_ROOT / "data" / "datasets"
    _CONFIGS_DIR = _PROJECT_ROOT / "configs"
    
    @classmethod
    def load(
        cls,
        dataset_name: str,
        num_samples: Optional[int] = None,
        seed: int = 42,
    ) -> List[DatasetExample]:
        """
        Load dataset examples.
        
        Args:
            dataset_name: One of the registered dataset names (see DATASET_REGISTRY)
            num_samples: Number of samples to draw (None = all)
            seed: Random seed for reproducible sampling
            
        Returns:
            List of DatasetExample dicts
        """
        logger.info(f"Loading dataset: {dataset_name} (n={num_samples}, seed={seed})")
        
        # Handle legacy name alias
        if dataset_name == "triviaqa":
            dataset_name = "trivia_qa"
        
        # Try registry-based loading first
        if dataset_name in DATASET_REGISTRY:
            examples = cls._load_from_registry(dataset_name)
        elif dataset_name == "sample":
            examples = cls._load_sample_questions()
        else:
            # Try to auto-discover the dataset
            examples = cls._load_auto_discover(dataset_name)
        
        # Sample if requested
        if num_samples is not None and num_samples < len(examples):
            rng = random.Random(seed)
            examples = rng.sample(examples, num_samples)
            logger.info(f"Sampled {num_samples} examples with seed={seed}")
        
        logger.info(f"Loaded {len(examples)} examples from '{dataset_name}'")
        return examples
    
    @classmethod
    def _load_from_registry(cls, dataset_name: str) -> List[DatasetExample]:
        """Load a dataset from the registry by name."""
        registry_entry = DATASET_REGISTRY[dataset_name]
        path = cls._DATASETS_DIR / registry_entry["file"]
        
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found at {path}. "
                f"Expected file: {registry_entry['file']}"
            )
        
        return cls._parse_json_dataset(path)
    
    @classmethod
    def _load_auto_discover(cls, dataset_name: str) -> List[DatasetExample]:
        """Auto-discover dataset by scanning data/datasets/{name}/ directory."""
        dataset_dir = cls._DATASETS_DIR / dataset_name
        
        if dataset_dir.is_dir():
            # Find the first JSON file in the directory
            json_files = list(dataset_dir.glob("*.json"))
            if json_files:
                logger.info(f"Auto-discovered dataset at {json_files[0]}")
                return cls._parse_json_dataset(json_files[0])
        
        # Final fallback: fail fast instead of hiding bad dataset configs
        logger.warning(f"Unknown dataset '{dataset_name}' requested. Failing fast.")
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    
    @classmethod
    def _parse_json_dataset(cls, path: Path) -> List[DatasetExample]:
        """Parse a standard JSON dataset file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        examples: List[DatasetExample] = []
        for item in data:
            examples.append(DatasetExample(
                id=item["id"],
                question=item["question"],
                answer=item["answer"],
                aliases=item.get("aliases", [item["answer"]]),
            ))
        
        return examples
    
    @classmethod
    def _load_sample_questions(cls) -> List[DatasetExample]:
        """Load built-in sample questions (Phase 2 fallback)."""
        path = cls._CONFIGS_DIR / "sample_questions.json"
        
        if not path.exists():
            raise FileNotFoundError(
                f"Sample questions not found at {path}. "
                "Please ensure configs/sample_questions.json exists."
            )
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        examples: List[DatasetExample] = []
        for item in data:
            examples.append(DatasetExample(
                id=item["id"],
                question=item["question"],
                answer=item["answer"],
                aliases=[item["answer"]],  # Sample questions don't have aliases
            ))
        
        return examples
    
    @classmethod
    def available_datasets(cls) -> List[dict]:
        """
        List available datasets with metadata.
        
        Returns:
            List of dicts with name, description, category, count, recommended_for.
        """
        datasets = []
        
        # Check all registered datasets
        for name, meta in DATASET_REGISTRY.items():
            path = cls._DATASETS_DIR / meta["file"]
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                datasets.append({
                    "name": name,
                    "display_name": meta["name"],
                    "description": meta["description"],
                    "category": meta["category"],
                    "total_examples": len(data),
                    "recommended_for": meta["recommended_for"],
                })
        
        # Check sample questions
        sample_path = cls._CONFIGS_DIR / "sample_questions.json"
        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                sample_data = json.load(f)
            datasets.append({
                "name": "sample",
                "display_name": "Sample Questions",
                "description": "Built-in smoke test questions",
                "category": "general",
                "total_examples": len(sample_data),
                "recommended_for": ["naive"],
            })
        
        return datasets
