# Prompt management: Langfuse comparison and LLMForge implementation

Reviewed on October 2, 2026 against the supplied Langfuse implementation notes, its official prompt-management documentation, and LLMForge's API, services, SDK, editor, and tests.

LLMForge already implements the central distinction correctly: immutable versions store content, while movable labels select what an application fetches. Saving a version never moves production unless the caller explicitly assigns that label. The improvements below bring the development workflow closer to Langfuse while retaining a small FastAPI/PostgreSQL application.

## Comparison

| Capability | Previous LLMForge behavior | Current behavior / decision |
| --- | --- | --- |
| Stable identity and immutable versions | Present; serialized allocation, stale-save protection, snapshot hashes | Retained. Configuration-only changes now create distinct snapshots too. Identical content/format/config is still rejected; rollback moves a label. |
| Text and chat prompts | Text only | Text templates or ordered system/user/assistant messages. Prompt type is fixed across versions. Compilation, playground, evaluations, HTTP fetch, and Python SDK preserve chat roles. |
| Variables | Mustache and restricted brace syntax | Variables are collected across chat messages as well as text. Substitution remains single-pass and never evaluates code. |
| Version configuration | Playground settings were transient | Bounded JSON object saved per version and returned as `prompt.config`. Applications decide how to use it. Playground generation settings remain explicit, initially using supported saved temperature/token values. |
| Labels | staging and production only | Custom case-sensitive labels plus automatic, read-only latest. Default fetch still selects production. Missing labels fail rather than silently selecting another version. |
| Tags | Absent | Prompt-level tags shared across versions, editable in Settings and filterable before pagination. API creation inherits tags when omitted on later versions. |
| Folder organization | Slash names rejected | Folder/name paths with valid segments, folder-prefix filtering, and SDK fetches for path names. This is namespace organization, without a separate folder-tree model. |
| Version comparison | Snapshot selection without content comparison | Reference/candidate content and config comparison in Versions. Matching leading/trailing lines are omitted. This is a bounded changed-region view rather than a full word-level diff engine. |
| Public/SDK creation | Read-only prompt SDK | Named creation and next-version creation share the same lifecycle service. Optional prompts:write keys can create versions and move labels; existing keys remain read-only. API-created versions record API as their source. |
| Fetch metadata | Version/text/template metadata | Chat messages, config, current tags and labels, version notes, content fingerprint, and current fetch name are returned. |
| Catalog API | Dashboard listing only | Project-key catalog listing with label/tag filters and accurate totals. |
| Persistence and concurrency | PostgreSQL with short transactions and prompt row locks | Retained. A version and its requested label assignments commit atomically. Concurrent first creation returns a conflict; explicit base_version detects stale updates. |
| Retrieval cache | SDK in-memory ETag cache; every fetch revalidates | Retained intentionally. This favors immediate label/key changes and easy-to-explain behavior over a Redis/TTL cache. No offline fallback or implicit retries. |
| Creator/audit trail | Timestamp and version notes | API/UI source added. User-level audit history and release-change event history remain extensions; source alone is not a full audit log. |

## Boundaries and remaining gaps

- **Message placeholders / tool messages:** ordinary role/content chat messages are supported. Runtime conversation-list placeholders, tool calls, and multimodal messages require a richer input contract and are not supported. Unknown roles are rejected.
- **Prompt composition:** references to other prompts, dependency graph validation, cycle/depth limits, and a resolution graph remain unimplemented. This is the most useful next advanced learning feature after the core lifecycle.
- **Trace linkage:** runs already reference immutable prompt versions, but this project has no general tracing product or per-prompt production trace analytics.
- **Protected labels and full auditing:** project owners control key creation; write-capable keys can move production. There is no configurable label protection or deployment approval policy.
- **Redis and change notifications:** no prompt cache epoch, webhook queue, or automation infrastructure was added. Direct database reads and synchronous writes remain sufficient for the learning project's current scale.
- **SDK breadth:** the Python client is implemented; a TypeScript SDK, TTL caching, fallback prompts, and automatic background refresh are separate extensions.
- **Legacy benchmarks:** text prompt adapters remain supported. Chat prompts use the normal Evaluations workspace; benchmark creation/execution rejects chat snapshots rather than flattening or ignoring them.

These are deliberate scope choices, not claims of full Langfuse parity. A useful next sequence is conversation placeholders, then composition with pinned dependencies, followed by release history/audit events if multiple users become relevant.

## Contracts and implementation map

- `backend/app/schemas/prompt.py`: shared text/chat content, bounded config, label and tag validation.
- `backend/app/services/prompt_service.py`: immutable version allocation, type invariant, content/config deduplication, atomic labels, shared catalog filtering.
- `backend/app/services/prompt_templates.py`: pure single-pass compilation and role-preserving chat compilation.
- `backend/app/api/prompt_library.py`: authenticated workspace operations.
- `backend/app/api/sdk_prompts.py`: project-key creation, listing, label assignment and fetching.
- `sdk/python/src/llmforge`: local compilation and project-key client methods.
- `frontend/src/components/prompts`: chat editor, saved configuration, comparison, releases, integration examples and explicit key capabilities.
- `m6n7o8p9q0r1_prompt_management.py`: additive migration preserving existing text versions, IDs and labels. Downgrade refuses to discard new capabilities while they are in use.

### Selector behavior

| Request | Result |
| --- | --- |
| No selector | production, or 404 if not assigned |
| version=3 | Saved v3, or 404 |
| label=latest | Newest saved version |
| label=experiment-a | Version assigned to that label, or 404 |
| Both version and label | 422 |
| Assign/remove latest | 422; automatic selector is read-only |

### Key capabilities

| Scope | Access |
| --- | --- |
| prompts:read | Fetch and list active prompts in the key's project |
| prompts:write | Create named prompts/versions and assign labels in that project |
| evaluations:write | Existing dataset/evaluation/CI endpoints |

Owner-created keys always include prompts:read. Adding a scope is explicit when creating a new key; existing keys are never upgraded automatically. Provider credentials are request-only. The SDK never retries mutation requests automatically.

## Verification

Tests cover config-only versions, canonical duplicate detection, chat compilation and provider role preservation, mock chat evaluation, custom/latest selection, missing selectors, write-scope enforcement, tenant isolation, tags inherited through API creation, and catalog totals before pagination. Frontend interactions cover chat ordering/configuration, invalid config, config-only saves, custom-label confirmation and saved-version comparison. Existing prompt, evaluation and SDK tests remain part of validation.

Completed validation: 566 backend tests, 26 Python SDK tests, and 47 frontend tests passed. ESLint and the frontend production build (including TypeScript checks) passed. The development database migration was applied successfully.

The authenticated Design review workspace contains `learning/support-chat` as a review example. The browser walkthrough verified ordered chat compilation in demo mode, a configuration-only second version, reference/candidate comparison, the custom `learning-preview` label, and combined `support` tag / `learning` folder filtering. Production was left unassigned. Desktop and 390-pixel mobile layouts were checked; screenshots are saved under `artifacts/prompt-management/`. Demo verification made no model calls.

## References

- Supplied Langfuse implementation notes: createPrompt, PromptService, public prompt handlers, labels/tags, dependency resolution and notification flow.
- [Langfuse version control](https://langfuse.com/docs/prompt-management/features/prompt-version-control): selectors, latest, label movement and version comparison.
- [Langfuse config](https://langfuse.com/docs/prompt-management/features/config): versioned application configuration.
- [Langfuse composability](https://langfuse.com/docs/prompt-management/features/composability): reusable prompt references; retained here as an advanced gap.
