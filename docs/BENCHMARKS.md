# Reasoning benchmarks

Benchmarks are LLMForge's separate experiment workflow. Use **Benchmarks** in the application to create a run, inspect samples and execution details, and compare two experiments. Prompt development and dataset evaluations are described in the [main README](../README.md); their results are not benchmark statistics.

## Experiment configuration

A benchmark combines a reasoning method (Naive, Chain-of-Thought, RAG, or ReAct), built-in dataset, model/provider, hyperparameters, and optional saved prompt version. Supported inference routes include Hugging Face, OpenRouter, Groq, and OpenAI-compatible endpoints. Adaptive routing can choose among configured providers using cost, latency, and error-rate telemetry; each run records the provider actually served, routing reason, and estimated cost.

The built-in dataset catalog includes `sample`, `trivia_qa`, `commonsense_qa`, `multi_hop`, `math_reasoning`, `react_bench`, `knowledge_base`, `prompt_injection`, `jailbreak`, and `edge_cases`. The last three are small diagnostic probes, not representative safety benchmarks.

## Results and comparisons

The experiment detail page exposes per-sample answers, references, routing information, retrieved context or agent traces when applicable, an effective execution manifest, latency and token/cost measures, and JSON/Markdown exports. The comparison workspace includes paired accuracy differences, bootstrap confidence intervals, McNemar's test, overlap and discordant counts, and methodology caveats. A pinned baseline can drive deterministic trajectory regression rules. Interpret significance and provider confounds together; a small or mismatched sample does not prove one strategy is better.

## Dispatch and recovery

PostgreSQL stores durable experiment status and partial results. The optional Upstash Redis/RQ dispatch path uses a worker heartbeat and circuit breaker; `auto` mode can fall back to inline FastAPI background execution. Inline work is best-effort and can be lost on an API restart. **Stop run** marks a queued/running attempt failed while preserving completed samples. A worker checks interruption between steps, so an in-flight provider call may finish. Attempt numbers prevent old queued jobs from starting a rerun.

For benchmark-specific setup, API calls, and tests, see the [testing guide](TESTING.md), the [backend README](../backend/README.md), and the relevant experiment API routes under `backend/app/api/`.
