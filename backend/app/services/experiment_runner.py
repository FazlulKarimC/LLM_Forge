"""Per-example execution and scoring for experiment attempts."""

import logging
import hashlib
import json
from decimal import Decimal, InvalidOperation
from typing import Any, List, Optional
from uuid import UUID

import numpy as np

from app.schemas.experiment import ExperimentResponse
from app.schemas.run_record import build_run_record
from app.services.benchmark_execution import run_blocking

logger = logging.getLogger(__name__)

SCORING_VERSION = "deterministic-v2"


def case_metadata(item):
    return {
        "case_fingerprint": hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest(),
        "scoring_version": SCORING_VERSION,
    }


def generation_metadata(item, result, config, *, cache_hit=False):
    return {**case_metadata(item), "cache_hit": cache_hit,
            "requested_provider": config.provider.value, "requested_model": config.model_name,
            "served_model": result.served_model, "usage_source": result.usage_source,
            "backend_provider": result.backend_provider,
            "cost_source": result.cost_source, "latency_kind": "http_wall_time"}


def cached_usage(result):
    # A cached response is evidence of a previous call, not a new billable call.
    from dataclasses import replace
    return replace(result, tokens_input=0, tokens_output=0, latency_ms=0, cost_usd=0)


class ExperimentRunExecutor:
    """Execute examples and build persisted run records."""

    def generation_seed(self, gen_config) -> Any:
        """Return the optional generation seed in a single place for cache calls."""
        return getattr(gen_config, "seed", None)

    def uses_robustness_scoring(self, dataset_name: str) -> bool:
        """Return whether the dataset should use deterministic safety scoring."""
        from app.services.dataset_service import DATASET_REGISTRY

        return DATASET_REGISTRY.get(dataset_name, {}).get("category") == "adversarial"

    def score_response(
        self,
        *,
        item,
        raw_output: str,
        parsed_answer: Optional[str],
        metrics_svc,
        use_robustness_scoring: bool,
    ) -> dict[str, Any]:
        """Score a response with QA metrics and optional adversarial robustness metadata."""
        aliases = item.get("aliases", [item["answer"]])
        is_exact, is_substring, f1_score, matched_alias = metrics_svc.check_any_alias_match(
            parsed_answer,
            aliases,
        )
        # Substring containment remains a diagnostic, never the primary QA score.
        is_correct = is_exact
        try:
            expected = Decimal(str(item["answer"]).replace(",", "").strip())
            predicted = Decimal(str(parsed_answer).replace(",", "").strip())
            if expected.is_finite() and predicted.is_finite():
                is_correct = expected == predicted
        except InvalidOperation:
            pass
        grader_results = None

        if use_robustness_scoring:
            from app.services.robustness_scorer import classify_response

            robustness = classify_response(
                raw_output or parsed_answer or "",
                expected_behavior=item.get("answer", "REFUSE"),
                category=item.get("category", ""),
            )
            if item.get("expected_behavior"):
                robustness["expected_behavior"] = item["expected_behavior"]
            grader_results = {"robustness": robustness}

            # Sentinel answers are evaluation directives, not strings the model should emit.
            is_correct = None if robustness.get("classification") == "inconclusive" else robustness.get("is_safe")
            f1_score = float(is_correct) if is_correct is not None else None

        return {
            "is_correct": is_correct,
            "score": f1_score,
            "is_exact_match": is_exact,
            "is_substring_match": is_substring,
            "match_alias": matched_alias,
            "grader_results": grader_results,
        }

    def build_prompt(
        self,
        question: str,
        reasoning_method: str,
        cot_examples,
        context_chunks,
        naive_prompt_template,
        cot_prompt_template,
        rag_prompt_template,
    ) -> str:
        """Build the final prompt for a single example."""
        if context_chunks:
            return rag_prompt_template.format(question, context_chunks)
        if reasoning_method == "cot":
            return cot_prompt_template.format(question, cot_examples)
        return naive_prompt_template.format(question)

    def parse_generation_output(
        self,
        raw_text: str,
        reasoning_method: str,
        *,
        used_rag: bool,
        naive_prompt_template,
        cot_prompt_template,
        rag_prompt_template,
    ) -> str:
        """Parse the model response using the matching prompt template."""
        if used_rag:
            return rag_prompt_template.parse_response(raw_text)
        if reasoning_method == "cot":
            return cot_prompt_template.parse_response(raw_text)
        return naive_prompt_template.parse_response(raw_text)

    def parse_response_with_method(
        self,
        raw_text: str,
        reasoning_method: str,
        *,
        used_rag: bool,
        naive_prompt_template,
        cot_prompt_template,
        rag_prompt_template,
    ) -> tuple:
        """Parse response and return (answer, parse_method) for confidence tracking."""
        if used_rag:
            return rag_prompt_template.parse_response_with_method(raw_text)
        if reasoning_method == "cot":
            return cot_prompt_template.parse_response_with_method(raw_text)
        return naive_prompt_template.parse_response_with_method(raw_text)

    def retrieve_rag_context(self, item, rag_config, rag_pipeline, profiler):
        """Retrieve RAG context and normalize it into run payload fields."""
        context_chunks = []
        retrieval_context = ""
        retrieved_chunk_payload = None

        if not rag_pipeline or not rag_config:
            return context_chunks, retrieval_context, retrieved_chunk_payload

        with profiler.section("rag_retrieval"):
            retrieval_result = rag_pipeline.retrieve(
                question=item["question"],
                method=rag_config.retrieval_method.value,
                top_k=rag_config.top_k,
            )

        context_chunks = [chunk.text for chunk in retrieval_result.chunks]
        # Zip chunks with scores from RetrievalResult — Chunk has no score
        # field, so getattr(chunk, "score", None) was silently returning None.
        chunk_scores = retrieval_result.scores if retrieval_result.scores else [None] * len(retrieval_result.chunks)
        retrieved_chunk_payload = {
            "chunks": [
                {
                    "text": chunk.text,
                    "chunk_id": chunk.id,
                    "title": chunk.title,
                    "score": score,
                }
                for chunk, score in zip(retrieval_result.chunks, chunk_scores)
            ]
        }
        retrieval_context = " ".join(context_chunks)
        logger.info(
            "[EXECUTE]   Retrieved %s chunks (%.0fms)",
            len(context_chunks),
            retrieval_result.latency_ms,
        )
        return context_chunks, retrieval_context, retrieved_chunk_payload

    def score_faithfulness(self, parsed_answer: str, retrieval_context: str, faithfulness_scorer, profiler):
        """Compute faithfulness only when the run used retrieved context."""
        if not retrieval_context or faithfulness_scorer is None:
            return None

        try:
            with profiler.section("faithfulness"):
                faithfulness = faithfulness_scorer.score(parsed_answer, retrieval_context)
            logger.info("[EXECUTE]   Faithfulness: %.3f", faithfulness)
            return faithfulness
        except Exception as exc:
            logger.warning("[EXECUTE]   Faithfulness scoring failed: %s", exc)
            return None

    def score_context_relevance(self, question: str, context_chunks, profiler):
        """Estimate context relevance from retrieved chunks for RAG runs."""
        if not context_chunks:
            return None

        try:
            with profiler.section("context_relevance"):
                from app.services.rag_service import CrossEncoderReranker as _CER

                reranker = _CER()
                scored = reranker.rerank(
                    question,
                    [
                        type("Chunk", (), {"id": f"c{index}", "text": text, "title": "", "index": index})()
                        for index, text in enumerate(context_chunks)
                    ],
                    top_k=len(context_chunks),
                )
                if scored:
                    return float(np.mean([score for _, score in scored]))
        except Exception as exc:
            logger.warning("[EXECUTE]   Context relevance scoring failed: %s", exc)

        return None

    def score_semantic_similarity(self, parsed_answer: Optional[str], expected_answer: Optional[str], profiler):
        """Compute cosine similarity between predicted and expected answers."""
        if not parsed_answer or not expected_answer:
            return None

        try:
            with profiler.section("semantic_similarity"):
                from app.services.rag_service import EmbeddingService as _ES

                emb_svc = _ES()
                embeddings = emb_svc.embed([parsed_answer, expected_answer])
                if len(embeddings) != 2:
                    return None

                norm_a = np.linalg.norm(embeddings[0])
                norm_b = np.linalg.norm(embeddings[1])
                if norm_a == 0 or norm_b == 0:
                    return 0.0

                cos_sim = float(np.dot(embeddings[0], embeddings[1]) / (norm_a * norm_b))
                return max(0.0, min(1.0, cos_sim))
        except Exception as exc:
            logger.warning("[EXECUTE]   Semantic similarity failed: %s", exc)
            return None

    async def build_agent_run_record(
        self,
        item,
        react_agent,
        react_prompt_template,
        profiler,
        metrics_svc,
        current_attempt: int,
        use_robustness_scoring: bool,
    ) -> dict[str, Any]:
        """Run a single example through the ReAct agent path."""
        with profiler.section("api_call"):
            log_start = len(getattr(react_agent._engine, "call_log", []))
            agent_result = await run_blocking(react_agent.run, item["question"], profiler)

        with profiler.section("parsing"):
            parsed_answer = react_prompt_template.parse_response(agent_result.answer)

        logger.info(
            "[EXECUTE]   Agent: %s iters, %s tool calls, success=%s (%s)",
            agent_result.total_iterations,
            agent_result.tool_calls,
            agent_result.success,
            agent_result.termination_reason,
        )

        with profiler.section("metrics"):
            score_result = self.score_response(
                item=item,
                raw_output=agent_result.answer,
                parsed_answer=parsed_answer,
                metrics_svc=metrics_svc,
                use_robustness_scoring=use_robustness_scoring,
            )

        agent_failure_mode = None
        agent_error_message = None
        if not agent_result.success:
            from app.models.run import FailureMode

            agent_failure_mode = FailureMode.UNKNOWN
            agent_error_message = f"agent_termination:{agent_result.termination_reason}"
            score_result["is_correct"] = False
            score_result["score"] = 0.0

        return build_run_record(
            example_id=item["id"],
            attempt=current_attempt,
            prompt=f"[Agent] {item['question']}",
            raw_output=agent_result.answer,
            expected_output=item["answer"],
            is_correct=score_result["is_correct"],
            score=score_result["score"],
            is_exact_match=score_result["is_exact_match"],
            is_substring_match=score_result["is_substring_match"],
            parsed_answer=parsed_answer,
            match_alias=score_result["match_alias"],
            tokens_input=agent_result.total_tokens_input,
            tokens_output=agent_result.total_tokens_output,
            latency_ms=agent_result.total_latency_ms,
            agent_trace=agent_result.trace_as_dict(),
            tool_calls=agent_result.tool_calls,
            failure_mode=agent_failure_mode,
            error_message=agent_error_message,
            grader_results=score_result["grader_results"],
            run_metadata={**case_metadata(item), "calls": getattr(react_agent._engine, "call_log", [])[log_start:], "latency_kind": "agent_wall_time"},
            cost_usd=sum(c.get("cost_usd") or 0 for c in getattr(react_agent._engine, "call_log", [])[log_start:]),
            served_provider=",".join(sorted({str(c.get("served_provider") or "unknown") for c in getattr(react_agent._engine, "call_log", [])[log_start:]})) or None,
        )

    async def build_standard_run_record(
        self,
        *,
        item,
        experiment_response: ExperimentResponse,
        reasoning_method: str,
        cot_examples,
        gen_config,
        max_tokens: int,
        engine,
        cache,
        profiler,
        metrics_svc,
        current_attempt: int,
        use_rag: bool,
        rag_config,
        rag_pipeline,
        faithfulness_scorer,
        naive_prompt_template,
        cot_prompt_template,
        rag_prompt_template,
        use_robustness_scoring: bool,
    ) -> dict[str, Any]:
        """Run one non-agent example through retrieval, generation, and scoring."""
        context_chunks = []
        retrieval_context = ""
        retrieved_chunk_payload = None
        if use_rag:
            context_chunks, retrieval_context, retrieved_chunk_payload = self.retrieve_rag_context(
                item,
                rag_config,
                rag_pipeline,
                profiler,
            )

        with profiler.section("prompt_build"):
            prompt = self.build_prompt(
                item["question"],
                reasoning_method,
                cot_examples,
                context_chunks,
                naive_prompt_template,
                cot_prompt_template,
                rag_prompt_template,
            )

        cache_seed = self.generation_seed(gen_config)
        result = None
        cache_hit = False
        if cache:
            with profiler.section("cache_lookup"):
                result = cache.get(
                    prompt,
                    experiment_response.config.model_name,
                    max_tokens,
                    gen_config.temperature,
                    cache_seed,
                )
            if result:
                cache_hit = True
                result = cached_usage(result)
                logger.info("[EXECUTE]   Cache HIT for example %s", item["id"])

        if result is None:
            with profiler.section("api_call"):
                result = await run_blocking(engine.generate, prompt, gen_config)

            if cache:
                cache.put(
                    prompt,
                    experiment_response.config.model_name,
                    max_tokens,
                    gen_config.temperature,
                    cache_seed,
                    result,
                )

        with profiler.section("parsing"):
            parsed_answer = self.parse_generation_output(
                result.text,
                reasoning_method,
                used_rag=use_rag,
                naive_prompt_template=naive_prompt_template,
                cot_prompt_template=cot_prompt_template,
                rag_prompt_template=rag_prompt_template,
            )

        faithfulness = self.score_faithfulness(
            parsed_answer,
            retrieval_context,
            faithfulness_scorer,
            profiler,
        )

        with profiler.section("metrics"):
            score_result = self.score_response(
                item=item,
                raw_output=result.text,
                parsed_answer=parsed_answer,
                metrics_svc=metrics_svc,
                use_robustness_scoring=use_robustness_scoring,
            )

        ctx_relevance = self.score_context_relevance(item["question"], context_chunks, profiler)
        # Optional remote scoring is not part of the bounded hosted profile.
        sem_sim = None

        return build_run_record(
            example_id=item["id"],
            attempt=current_attempt,
            prompt=prompt,
            raw_output=result.text,
            expected_output=item["answer"],
            is_correct=score_result["is_correct"],
            score=score_result["score"],
            is_exact_match=score_result["is_exact_match"],
            is_substring_match=score_result["is_substring_match"],
            parsed_answer=parsed_answer,
            match_alias=score_result["match_alias"],
            semantic_similarity=sem_sim,
            tokens_input=result.tokens_input,
            tokens_output=result.tokens_output,
            latency_ms=result.latency_ms,
            gpu_memory_mb=result.gpu_memory_mb,
            faithfulness_score=faithfulness,
            retrieved_chunks=retrieved_chunk_payload,
            context_relevance_score=ctx_relevance,
            served_provider=result.served_provider,
            routing_reason=result.routing_reason,
            cost_usd=result.cost_usd,
            failure_mode=result.failure_mode,
            error_message=result.error_message,
            grader_results=score_result["grader_results"],
            run_metadata=generation_metadata(item, result, experiment_response.config, cache_hit=cache_hit),
        )

    async def flush_runs(self, run_service, experiment_id: UUID, runs_batch_data: List[dict[str, Any]], *, force: bool = False):
        """Flush buffered run rows to the database in consistent batch sizes."""
        if not runs_batch_data:
            return
        if not force and len(runs_batch_data) < 50:
            return

        await run_service.create_runs_batch(experiment_id, runs_batch_data)
        runs_batch_data.clear()

    async def execute_batched_runs(
        self,
        *,
        experiment_id: UUID,
        experiment_response: ExperimentResponse,
        examples,
        reasoning_method: str,
        cot_examples,
        gen_config,
        max_tokens: int,
        engine,
        cache,
        profiler,
        metrics_svc,
        run_service,
        current_attempt: int,
        batch_size: int,
        naive_prompt_template,
        cot_prompt_template,
        use_robustness_scoring: bool,
        ensure_active=None,
        persist_cases=None,
    ) -> dict[str, int]:
        """Execute the non-RAG non-agent fast path using batched generation."""
        logger.info("[EXECUTE] Using batched execution (batch_size=%s)", batch_size)
        batch_stats = {"batches_processed": 0, "total_prompts_batched": 0}
        cache_seed = self.generation_seed(gen_config)

        for batch_start in range(0, len(examples), batch_size):
            if ensure_active:
                await ensure_active()
            batch_end = min(batch_start + batch_size, len(examples))
            batch_items = examples[batch_start:batch_end]
            logger.info(
                "[EXECUTE] Batch %s: examples %s-%s",
                batch_start // batch_size + 1,
                batch_start + 1,
                batch_end,
            )

            prompts = []
            cached_results = {}
            uncached_indices = []

            with profiler.section("prompt_build"):
                for local_idx, item in enumerate(batch_items):
                    if reasoning_method == "cot":
                        prompt = cot_prompt_template.format(item["question"], cot_examples)
                    else:
                        prompt = naive_prompt_template.format(item["question"])
                    prompts.append(prompt)

                    if not cache:
                        uncached_indices.append(local_idx)
                        continue

                    with profiler.section("cache_lookup"):
                        cached = cache.get(
                            prompt,
                            experiment_response.config.model_name,
                            max_tokens,
                            gen_config.temperature,
                            cache_seed,
                        )
                    if cached:
                        cached_results[local_idx] = cached
                        logger.info("[EXECUTE]   Cache HIT for example %s", batch_start + local_idx + 1)
                    else:
                        uncached_indices.append(local_idx)

            uncached_prompts = [prompts[idx] for idx in uncached_indices]
            batch_gen_results = []
            if uncached_prompts:
                with profiler.section("api_call"):
                    batch_gen_results = await run_blocking(
                        engine.generate_batch,
                        uncached_prompts,
                        gen_config,
                    )
                if ensure_active:
                    await ensure_active()

                if cache:
                    for uncached_idx, gen_result in zip(uncached_indices, batch_gen_results):
                        cache.put(
                            prompts[uncached_idx],
                            experiment_response.config.model_name,
                            max_tokens,
                            gen_config.temperature,
                            cache_seed,
                            gen_result,
                        )

            gen_results_iterator = iter(batch_gen_results)
            all_results = []
            for local_idx in range(len(batch_items)):
                if local_idx in cached_results:
                    all_results.append(cached_usage(cached_results[local_idx]))
                else:
                    all_results.append(next(gen_results_iterator))

            runs_batch_data = []
            for local_idx, (item, result) in enumerate(zip(batch_items, all_results)):
                with profiler.section("parsing"):
                    if reasoning_method == "cot":
                        parsed_answer, parse_method = cot_prompt_template.parse_response_with_method(result.text)
                    else:
                        parsed_answer, parse_method = naive_prompt_template.parse_response_with_method(result.text)

                with profiler.section("metrics"):
                    score_result = self.score_response(
                        item=item,
                        raw_output=result.text,
                        parsed_answer=parsed_answer,
                        metrics_svc=metrics_svc,
                        use_robustness_scoring=use_robustness_scoring,
                    )

                runs_batch_data.append(
                    build_run_record(
                        example_id=item["id"],
                        attempt=current_attempt,
                        prompt=prompts[local_idx],
                        raw_output=result.text,
                        expected_output=item["answer"],
                        is_correct=score_result["is_correct"],
                        score=score_result["score"],
                        is_exact_match=score_result["is_exact_match"],
                        is_substring_match=score_result["is_substring_match"],
                        parsed_answer=parsed_answer,
                        match_alias=score_result["match_alias"],
                        tokens_input=result.tokens_input,
                        tokens_output=result.tokens_output,
                        latency_ms=result.latency_ms,
                        gpu_memory_mb=result.gpu_memory_mb,
                        served_provider=result.served_provider,
                        routing_reason=result.routing_reason,
                        cost_usd=result.cost_usd,
                        failure_mode=result.failure_mode,
                        error_message=result.error_message,
                        grader_results=score_result["grader_results"],
                        run_metadata={"parse_method": parse_method, **generation_metadata(item, result, experiment_response.config, cache_hit=local_idx in cached_results)},
                    )
                )

            if runs_batch_data:
                if ensure_active:
                    await ensure_active()
                if persist_cases:
                    await persist_cases(runs_batch_data)
                else:
                    await run_service.create_runs_batch(experiment_id, runs_batch_data)

            batch_stats["batches_processed"] += 1
            batch_stats["total_prompts_batched"] += len(batch_items)

        return batch_stats

    async def execute_sequential_runs(
        self,
        *,
        experiment_id: UUID,
        experiment_response: ExperimentResponse,
        examples,
        reasoning_method: str,
        cot_examples,
        gen_config,
        max_tokens: int,
        engine,
        cache,
        profiler,
        metrics_svc,
        run_service,
        current_attempt: int,
        use_batching: bool,
        use_rag: bool,
        use_robustness_scoring: bool,
        rag_config,
        rag_pipeline,
        faithfulness_scorer,
        react_agent,
        naive_prompt_template,
        cot_prompt_template,
        rag_prompt_template,
        react_prompt_template,
        ensure_active=None,
        persist_cases=None,
    ) -> dict[str, int]:
        """Execute the sequential path used by RAG and agent experiments."""
        if use_batching:
            logger.info("[EXECUTE] Batching disabled for RAG/agent execution")

        runs_batch_data: List[dict[str, Any]] = []
        for index, item in enumerate(examples):
            if ensure_active:
                await ensure_active()
            logger.info("[EXECUTE] Processing %s/%s: %s", index + 1, len(examples), item["id"])

            if reasoning_method == "react" and react_agent is not None:
                run_record = await self.build_agent_run_record(
                    item,
                    react_agent,
                    react_prompt_template,
                    profiler,
                    metrics_svc,
                    current_attempt,
                    use_robustness_scoring,
                )
            else:
                run_record = await self.build_standard_run_record(
                    item=item,
                    experiment_response=experiment_response,
                    reasoning_method=reasoning_method,
                    cot_examples=cot_examples,
                    gen_config=gen_config,
                    max_tokens=max_tokens,
                    engine=engine,
                    cache=cache,
                    profiler=profiler,
                    metrics_svc=metrics_svc,
                    current_attempt=current_attempt,
                    use_rag=use_rag,
                    rag_config=rag_config,
                    rag_pipeline=rag_pipeline,
                    faithfulness_scorer=faithfulness_scorer,
                    naive_prompt_template=naive_prompt_template,
                    cot_prompt_template=cot_prompt_template,
                    rag_prompt_template=rag_prompt_template,
                    use_robustness_scoring=use_robustness_scoring,
                )

            if ensure_active:
                await ensure_active()
            if persist_cases:
                await persist_cases([run_record])
            else:
                runs_batch_data.append(run_record)
                await self.flush_runs(run_service, experiment_id, runs_batch_data)

        if ensure_active:
            await ensure_active()
        await self.flush_runs(run_service, experiment_id, runs_batch_data, force=True)
        return {"batches_processed": 0, "total_prompts_batched": 0}
