"""
Dataset Service

Abstracts dataset loading, sampling, and caching.
Supports curated TriviaQA data and fallback sample questions.
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


class DatasetService:
    """
    Service for loading evaluation datasets.
    
    Supports:
    - trivia_qa: Curated TriviaQA-style questions (100 questions)
    - sample: Built-in sample questions (10 questions)
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
            dataset_name: One of "trivia_qa", "sample"
            num_samples: Number of samples to draw (None = all)
            seed: Random seed for reproducible sampling
            
        Returns:
            List of DatasetExample dicts
        """
        logger.info(f"Loading dataset: {dataset_name} (n={num_samples}, seed={seed})")
        
        if dataset_name in ("trivia_qa", "triviaqa"):
            examples = cls._load_triviaqa()
        elif dataset_name == "sample":
            examples = cls._load_sample_questions()
        else:
            # Try trivia_qa as default
            logger.warning(f"Unknown dataset '{dataset_name}', falling back to sample questions")
            examples = cls._load_sample_questions()
        
        # Sample if requested
        if num_samples is not None and num_samples < len(examples):
            rng = random.Random(seed)
            examples = rng.sample(examples, num_samples)
            logger.info(f"Sampled {num_samples} examples with seed={seed}")
        
        logger.info(f"Loaded {len(examples)} examples from '{dataset_name}'")
        return examples
    
    @classmethod
    def _load_triviaqa(cls) -> List[DatasetExample]:
        """Load curated TriviaQA dataset from local JSON."""
        path = cls._DATASETS_DIR / "triviaqa" / "trivia_qa.json"
        
        if not path.exists():
            raise FileNotFoundError(
                f"TriviaQA dataset not found at {path}. "
                "Please ensure data/datasets/triviaqa/trivia_qa.json exists."
            )
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate format
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
        """List available datasets with metadata."""
        datasets = []
        
        # Check TriviaQA
        tqa_path = cls._DATASETS_DIR / "triviaqa" / "trivia_qa.json"
        if tqa_path.exists():
            with open(tqa_path, "r", encoding="utf-8") as f:
                tqa_data = json.load(f)
            datasets.append({
                "name": "trivia_qa",
                "description": "Curated TriviaQA-style questions",
                "total_examples": len(tqa_data),
            })
        
        # Check sample questions
        sample_path = cls._CONFIGS_DIR / "sample_questions.json"
        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                sample_data = json.load(f)
            datasets.append({
                "name": "sample",
                "description": "Built-in sample questions (Phase 2)",
                "total_examples": len(sample_data),
            })
        
        return datasets
