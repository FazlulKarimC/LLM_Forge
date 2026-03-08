"""
LLM-as-Judge Service (P2 #13)

Budget-capped, sampled LLM evaluation for metrics that require
generative assessment (coherence, helpfulness, factuality).

Uses free HF Inference API with a small instruct model.
Only evaluates a random subset of runs to stay within free-tier limits.
"""

import asyncio
import json
import logging
import random
from typing import List, Optional, Dict, Any
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.run import Run
from app.services.inference.base import GenerationConfig
from app.services.inference.hf_api_engine import HFAPIEngine

logger = logging.getLogger(__name__)

# Budget defaults: max 20 judge calls per experiment
DEFAULT_SAMPLE_SIZE = 20
MAX_SAMPLE_SIZE = 50

# Evaluation dimensions
JUDGE_DIMENSIONS = ["coherence", "helpfulness", "factuality"]

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. Rate the following AI response on a scale of 1-5.

**Question:** {question}
**Expected Answer:** {expected}
**AI Response:** {response}

Rate on these dimensions (1=very poor, 5=excellent):
1. **Coherence**: Is the response well-structured, grammatically correct, and logically consistent?
2. **Helpfulness**: Does the response address the question and provide useful information?
3. **Factuality**: Is the response factually accurate compared to the expected answer?

Respond ONLY in this exact JSON format:
{{"coherence": <1-5>, "helpfulness": <1-5>, "factuality": <1-5>}}"""


class LLMJudgeService:
    """
    Service for LLM-as-a-judge evaluation on sampled subsets.
    
    Uses HF Inference API with a free instruct model to evaluate
    generated responses on coherence, helpfulness, and factuality.
    Budget-capped to prevent runaway API costs.
    """
    
    def __init__(
        self,
        db: AsyncSession,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        sample_size: int = DEFAULT_SAMPLE_SIZE,
    ):
        self.db = db
        self.model_id = model_id
        self.sample_size = min(sample_size, MAX_SAMPLE_SIZE)
    
    async def evaluate_experiment(
        self,
        experiment_id: UUID,
        attempt: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a sample of runs from an experiment using LLM-as-judge.
        
        Args:
            experiment_id: UUID of the experiment to evaluate
            attempt: Specific attempt to evaluate (default: latest)
            
        Returns:
            Dictionary with per-dimension scores and metadata
        """
        query = select(Run).where(Run.experiment_id == experiment_id)
        result = await self.db.execute(query)
        all_runs = result.scalars().all()
        
        if not all_runs:
            return {"error": "No runs found", "scores": {}}
        
        if attempt is None:
            attempt = max(r.attempt for r in all_runs)
        runs = [r for r in all_runs if r.attempt == attempt]
        
        if len(runs) > self.sample_size:
            runs = random.sample(runs, self.sample_size)
        
        logger.info(
            "[LLM-JUDGE] Evaluating %s sampled runs from experiment %s (attempt %s)",
            len(runs),
            experiment_id,
            attempt,
        )

        try:
            judge_engine = HFAPIEngine(model_name=self.model_id)
        except ValueError:
            logger.warning("[LLM-JUDGE] HF_TOKEN not set, skipping judge evaluation")
            return {
                "error": "HF_TOKEN not set",
                "scores": {},
                "sample_size": len(runs),
                "evaluated": 0,
                "attempt": attempt,
            }
        
        judgments: List[Dict[str, int]] = []
        evaluated_ids = []

        try:
            for run in runs:
                if not run.raw_output or not run.expected_output:
                    continue
                
                try:
                    judgment = await self._judge_single(
                        question=run.prompt,
                        expected=run.expected_output,
                        response=run.raw_output,
                        engine=judge_engine,
                    )
                    if judgment:
                        judgments.append(judgment)
                        evaluated_ids.append(str(run.id))
                except Exception as e:
                    logger.warning("[LLM-JUDGE] Failed to judge run %s: %s", run.id, e)
                    continue
        finally:
            judge_engine.unload_model()
        
        if not judgments:
            return {
                "error": "No successful judgments",
                "scores": {},
                "sample_size": len(runs),
                "evaluated": 0,
                "attempt": attempt,
            }
        
        aggregated = {}
        for dim in JUDGE_DIMENSIONS:
            values = [j.get(dim, 0) for j in judgments if dim in j]
            if values:
                arr = np.array(values, dtype=float)
                aggregated[dim] = {
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "std": float(np.std(arr)),
                    "count": len(values),
                }
        
        return {
            "model_judge": self.model_id,
            "sample_size": len(runs),
            "evaluated": len(judgments),
            "attempt": attempt,
            "scores": aggregated,
            "evaluated_run_ids": evaluated_ids,
            "method": "llm_as_judge",
            "budget_cap": self.sample_size,
        }
    
    async def _judge_single(
        self,
        question: str,
        expected: str,
        response: str,
        engine: HFAPIEngine,
    ) -> Optional[Dict[str, int]]:
        """
        Judge a single (question, expected, response) triple.
        
        Returns dict with coherence, helpfulness, factuality scores (1-5).
        """
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question[:500],
            expected=expected[:500],
            response=response[:500],
        )

        generation = await asyncio.to_thread(
            engine.generate,
            prompt,
            GenerationConfig(
                max_tokens=100,
                temperature=0.1,
                top_p=0.9,
            ),
        )

        if generation.error_message or not generation.text:
            logger.warning("[LLM-JUDGE] Judge generation failed: %s", generation.error_message or generation.finish_reason)
            return None

        json_start = generation.text.find("{")
        json_end = generation.text.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            return None

        try:
            parsed = json.loads(generation.text[json_start:json_end])
        except json.JSONDecodeError as e:
            logger.warning("[LLM-JUDGE] Failed to parse judge response: %s", e)
            return None
        
        result = {}
        for dim in JUDGE_DIMENSIONS:
            val = parsed.get(dim)
            if isinstance(val, (int, float)) and 1 <= val <= 5:
                result[dim] = int(val)
        
        return result if result else None
