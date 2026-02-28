"""
Run Service

Service for logging individual LLM inference runs.
Each run represents a single LLM call with all its metadata.
"""

from uuid import UUID
from typing import Optional

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.run import Run


class RunService:
    """
    Service for managing inference run logs.
    
    Logs each LLM call to the database with full metadata:
    - Input/output text
    - Token counts
    - Latency
    - Correctness (if ground truth available)
    """
    
    def __init__(self, db: AsyncSession):
        """
        Initialize with database session.
        
        Args:
            db: Async database session
        """
        self.db = db
    
    async def create_run(
        self,
        experiment_id: UUID,
        input_text: str,
        output_text: str,
        tokens_input: int,
        tokens_output: int,
        latency_ms: float,
        example_id: Optional[str] = None,
        expected_output: Optional[str] = None,
        is_correct: Optional[bool] = None,
        score: Optional[float] = None,
        is_exact_match: Optional[bool] = None,
        is_substring_match: Optional[bool] = None,
        parsed_answer: Optional[str] = None,
        match_alias: Optional[str] = None,
        semantic_similarity: Optional[float] = None,
        gpu_memory_mb: Optional[float] = None,
        agent_trace: Optional[dict] = None,
        tool_calls: Optional[int] = None,
        faithfulness_score: Optional[float] = None,
        retrieved_chunks: Optional[dict] = None,
        context_relevance_score: Optional[float] = None,
        attempt: int = 1,
    ) -> Run:
        """
        Create a new run log entry.
        
        Args:
            experiment_id: Parent experiment UUID
            input_text: Full prompt sent to model
            output_text: Generated response
            tokens_input: Input token count
            tokens_output: Output token count
            latency_ms: Generation time in milliseconds
            example_id: Dataset example ID (optional)
            expected_output: Ground truth answer (optional)
            is_correct: Whether answer was correct (optional)
            score: Soft score 0-1 (optional)
            is_exact_match: Whether prediction exactly matches ground truth (optional)
            is_substring_match: Whether ground truth is substring of prediction (optional)
            parsed_answer: Extracted answer from model output (optional)
            match_alias: Which alias was matched (optional)
            semantic_similarity: Embedding cosine similarity (optional)
            gpu_memory_mb: GPU memory usage (optional)
            agent_trace: Full agent Thought/Action/Observation trace (optional)
            tool_calls: Number of tool invocations (optional)
            faithfulness_score: RAG faithfulness evaluation (optional)
            retrieved_chunks: RAG retrieved documents (optional)
            context_relevance_score: CrossEncoder context relevance (optional)
            attempt: Attempt number for non-destructive re-runs
        
        Returns:
            Created Run instance
        """
        run = Run(
            experiment_id=experiment_id,
            example_id=example_id,
            input_text=input_text,
            output_text=output_text,
            expected_output=expected_output,
            is_correct=is_correct,
            score=score,
            is_exact_match=is_exact_match,
            is_substring_match=is_substring_match,
            parsed_answer=parsed_answer,
            match_alias=match_alias,
            semantic_similarity=semantic_similarity,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            gpu_memory_mb=gpu_memory_mb,
            agent_trace=agent_trace,
            tool_calls=tool_calls,
            faithfulness_score=faithfulness_score,
            retrieved_chunks=retrieved_chunks,
            context_relevance_score=context_relevance_score,
            attempt=attempt,
        )
        
        self.db.add(run)
        await self.db.flush()
        
        return run

    async def create_runs_batch(
        self,
        experiment_id: UUID,
        runs_data: list[dict],
    ) -> list[Run]:
        """
        Create multiple run log entries efficiently.
        
        Args:
            experiment_id: Parent experiment UUID
            runs_data: List of dictionaries containing run metadata
            
        Returns:
            List of created Run instances
        """
        runs = [Run(experiment_id=experiment_id, **data) for data in runs_data]
        self.db.add_all(runs)
        await self.db.flush()
        return runs

    async def clear_runs(self, experiment_id: UUID) -> None:
        """
        Delete all run logs for a given experiment.
        
        Useful when re-running an experiment to clear old data.
        
        Args:
            experiment_id: Parent experiment UUID
        """
        await self.db.execute(
            delete(Run).where(Run.experiment_id == experiment_id)
        )
        await self.db.flush()

