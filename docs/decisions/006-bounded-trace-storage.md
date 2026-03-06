# ADR 006: Bounded Trace Storage for Free Tier DB Limits

**Status**: Accepted  
**Date**: 2024-07-10

### Context
A core feature of LLMForge is deep debuggability — we want users to click a run and see the exact prompt sent to the LLM, the exact tool calls made by the agent, and the raw text output before parsing. 

Storing this data (the `trace_data` JSONB column on the `Run` model) means writing up to 10KB+ of payload per dataset example. If a user runs 10 experiments of 100 questions each, generating massive `agent_trace` payloads, a standard Supabase or NeonDB free-tier instance (which typically caps at 256MB–500MB) will rapidly exhaust its disk quota.

### Decision
We implement **Sampled Trace Capture with Soft TTLs**.
The `ExperimentService` restricts full trace captures in the database.
1. **Always capture failed runs**: The entire point of a trace is debugging. If `is_correct` evaluation fails, we keep the trace.
2. **Sample successful runs**: If the evaluation succeeds, store the trace for only a random sample (e.g., 5 runs per experiment). Nullify the JSONB column for the rest.
3. **Implicit TTL via Cascading Deletion**: Soft deleting an experiment automatically deletes its orphaned runs and traces.

### Consequences
- **Positive:** Drastically bounded database growth curve. Projects can theoretically run 10x more experiments before hitting DB limits.
- **Positive:** Minimizes network payload sizes over the wire.
- **Negative:** Users cannot inspect the "perfect" runs unless they happen to fall in the random inclusion sample.

### Constraints Addressed
- Serverless Postgres Free Tier limits (primarily disk space exhaustion).
