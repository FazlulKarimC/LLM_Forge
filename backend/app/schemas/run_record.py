"""
Run Record Contract

TypedDict defining the exact shape of per-run dicts built by
ExperimentRunExecutor and consumed by RunService.create_runs_batch().

Every dict passed to create_runs_batch() is unpacked into Run(**data),
so these keys must match Run model column names exactly.
"""

from typing import Optional, TypedDict

from app.models.run import FailureMode


class RunRecordDict(TypedDict, total=False):
    """Typed contract for run record dicts flowing through the execution pipeline.

    Required keys are set via ``total=False`` + explicit construction
    in ``build_run_record`` to keep the TypedDict flexible for partial
    construction (e.g. agent path omits RAG fields).
    """

    # ── Identity ──
    example_id: str
    attempt: int

    # ── Input / Output ──
    prompt: str
    raw_output: Optional[str]
    expected_output: Optional[str]

    # ── Evaluation ──
    is_correct: bool
    score: float
    is_exact_match: bool
    is_substring_match: bool
    parsed_answer: Optional[str]
    match_alias: Optional[str]

    # ── Performance ──
    tokens_input: Optional[int]
    tokens_output: Optional[int]
    latency_ms: Optional[float]
    gpu_memory_mb: Optional[float]

    # ── RAG-specific ──
    faithfulness_score: Optional[float]
    retrieved_chunks: Optional[dict]
    context_relevance_score: Optional[float]
    semantic_similarity: Optional[float]

    # ── Agent-specific ──
    agent_trace: Optional[dict]
    tool_calls: Optional[int]

    # ── Failure tracking ──
    failure_mode: Optional[FailureMode]
    error_message: Optional[str]

    # ── Grading ──
    grader_results: Optional[dict]

    # ── Routing telemetry ──
    served_provider: Optional[str]
    routing_reason: Optional[str]
    cost_usd: Optional[float]

    # ── Audit metadata ──
    run_metadata: Optional[dict]


def build_run_record(
    *,
    example_id: str,
    attempt: int,
    prompt: str,
    raw_output: Optional[str],
    expected_output: Optional[str],
    is_correct: bool,
    score: float,
    is_exact_match: bool,
    is_substring_match: bool,
    parsed_answer: Optional[str],
    match_alias: Optional[str],
    tokens_input: Optional[int] = None,
    tokens_output: Optional[int] = None,
    latency_ms: Optional[float] = None,
    gpu_memory_mb: Optional[float] = None,
    grader_results: Optional[dict] = None,
    # RAG fields
    faithfulness_score: Optional[float] = None,
    retrieved_chunks: Optional[dict] = None,
    context_relevance_score: Optional[float] = None,
    semantic_similarity: Optional[float] = None,
    # Agent fields
    agent_trace: Optional[dict] = None,
    tool_calls: Optional[int] = None,
    # Failure tracking
    failure_mode: Optional[FailureMode] = None,
    error_message: Optional[str] = None,
    # Routing
    served_provider: Optional[str] = None,
    routing_reason: Optional[str] = None,
    cost_usd: Optional[float] = None,
    # Audit
    run_metadata: Optional[dict] = None,
) -> RunRecordDict:
    """Build a typed run record dict for RunService.create_runs_batch().

    Centralises the field contract so that agent, standard, and batched
    execution paths all produce structurally identical records.  Adding
    a new field here automatically surfaces type errors at every call site.
    """
    record: RunRecordDict = {
        "example_id": example_id,
        "attempt": attempt,
        "prompt": prompt,
        "raw_output": raw_output,
        "expected_output": expected_output,
        "is_correct": is_correct,
        "score": score,
        "is_exact_match": is_exact_match,
        "is_substring_match": is_substring_match,
        "parsed_answer": parsed_answer,
        "match_alias": match_alias,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "latency_ms": latency_ms,
        "gpu_memory_mb": gpu_memory_mb,
        "grader_results": grader_results,
        "failure_mode": failure_mode,
        "error_message": error_message,
    }

    # Attach optional domain-specific fields only when present,
    # keeping the dict sparse for non-RAG / non-agent paths.
    if faithfulness_score is not None:
        record["faithfulness_score"] = faithfulness_score
    if retrieved_chunks is not None:
        record["retrieved_chunks"] = retrieved_chunks
    if context_relevance_score is not None:
        record["context_relevance_score"] = context_relevance_score
    if semantic_similarity is not None:
        record["semantic_similarity"] = semantic_similarity
    if agent_trace is not None:
        record["agent_trace"] = agent_trace
    if tool_calls is not None:
        record["tool_calls"] = tool_calls
    if served_provider is not None:
        record["served_provider"] = served_provider
    if routing_reason is not None:
        record["routing_reason"] = routing_reason
    if cost_usd is not None:
        record["cost_usd"] = cost_usd
    if run_metadata is not None:
        record["run_metadata"] = run_metadata

    return record
