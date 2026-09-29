# UI and README review plan

Reviewed 2026-09-29 against commit `fd85019`. Implementation began on `codex/ui-readme-refresh`; this document preserves the review rationale and acceptance criteria.

## Review basis and limits

Inspected the application shell, dashboard, prompt catalog/editor/playground, dataset editor, evaluation setup/history/results, settings, workspace switching, shared styles, landing page, public docs, root README, SDK README, and relevant API list responses. The working tree was clean when the review began.

A saved browser permission blocked localhost access. Findings about navigation, component structure, copy, and CSS are source-verified. Actual rendered spacing, contrast, responsive behavior, keyboard interaction, and authenticated flows still need browser validation. No production records were changed or model calls made. The temporary frontend development server was stopped after browser access was rejected. Tests were not rerun for this documentation-only review.

## Main conclusion

The product has evolved into a prompt development and regression evaluation workspace, while substantial parts of its presentation still describe and prioritize a reasoning benchmark console. Changing the navigation, screen hierarchy, and connections between existing features will make a larger difference than changing colors alone.

Use one product description consistently: **LLMForge — version, evaluate, and release prompts.** The primary journey should be visible throughout the application:

**Prompt draft → playground → saved version → dataset evaluation → comparison → release label → Python SDK / CI.**

Keep reasoning benchmarks as a supported secondary workflow. Keep their statistical comparison and execution diagnostics distinct from prompt evaluation results.

## Evidence and proposed changes

| Priority | Finding and evidence | Proposed change |
| --- | --- | --- |
| P0 | `frontend/src/components/app-shell.tsx:36` puts Settings before Overview and mixes Benchmarks, Compare, and New experiment with the prompt workflow. The command palette at line 137 only covers older benchmark routes. | Group navigation around the primary workflow; place Settings and Docs in a utility group; make benchmark comparison contextual; add prompt/dataset/evaluation commands. Label the palette as navigation/commands until it actually searches records. |
| P0 | `frontend/src/app/(app)/dashboard/page.tsx:164` derives all four summary cards from benchmark stats. Its primary buttons are Browse experiments and New experiment. Demo setup is a separate panel. | Make the overview show recent evaluations, prompt releases, and a clear first evaluation action. Move benchmark stats to Benchmarks. Keep readiness as a compact status with expandable diagnostics. |
| P0 | `frontend/src/components/evaluations-workbench.tsx:192` opens the long creation form before history; history at line 604 is a pair of dropdowns. | Make a run table the default Evaluations view, with a New evaluation action opening setup. Give a selected run a dedicated results view. Existing URL selection restoration should be preserved and extended. |
| P1 | `frontend/src/components/prompts/prompt-workbench.tsx:102` mixes metadata, editor, history, playground, releases, curl instructions, and archive controls in two vertical stacks. | Keep editing and testing together; move history, releases, integration, and settings into distinct tabs or secondary panels. Put Save version and Evaluate saved version in a persistent toolbar. |
| P1 | `frontend/src/components/datasets-workbench.tsx:269` uses a JSON textarea as the main case editor. The Evaluate prompts link points only to `/evaluations`. | Make a case table and row editor the normal interaction; keep JSON as an advanced editor. Carry the selected dataset and saved revision into evaluation setup. |
| P1 | `frontend/src/components/evaluations/evaluation-results.tsx` already calculates improved/regressed counts and guards differing assertions, but renders full inputs and outputs inside a wide table. | Add compact summary metrics, explicit reference/candidate labels, regression filters, and a case details panel. Preserve dataset revision compatibility checks and differing-assertion warnings. |
| P1 | `frontend/src/app/globals.css:143` gives headers large decorated panels; titles scale to 3.5rem. Content is capped at 1180px. Dataset/evaluation pages use separate heading and spacing conventions. | Introduce compact application headers, shared fields/toolbars/tables, calmer surfaces, and a wide workbench layout. Scope changes so marketing page typography is independent. |
| P1 | Root `README.md:25` introduces the new journey, but line 41 still calls reasoning strategy comparison the core question. Most Key Features, the architecture diagram, and Running an Experiment describe benchmarks. | Rewrite the root README around the main prompt workflow, with a short secondary benchmark section and links to detailed benchmark documentation. |
| P2 | Settings combines account, organization/project creation, and SDK keys; SDK fetch instructions are embedded in the prompt release panel. | Group Settings into Project/API keys and Workspace/account sections, and provide contextual Python and curl examples from a prompt's Integrate tab. Preserve current permissions and one-time key display. |
| P2 | Landing copy and metadata already emphasize prompts and evaluations. The README still describes a live comparison preview on `/`, while the current page contains explanatory workflow cards. | Preserve the improved positioning. Add a real product screenshot or short walkthrough and correct the route description. Avoid rewriting already accurate copy solely for novelty. |

The shell displays `G D`, `G E`, etc. as shortcut badges, but the inspected handler implements Ctrl/Cmd+K and Escape. Either implement the displayed shortcuts with safeguards for text entry or remove those badges. This is a small source-verified consistency issue to address with the shell work.

## Relevant product references

- [Langfuse Playground](https://langfuse.com/docs/prompt-management/features/playground): prompt-to-playground transitions, saving an iteration, and viewing alternatives together. Adapt the connected workflow; LLMForge already supports multiple provider targets for one draft, so expose that existing capability before adding multiple independent prompt drafts.
- [Langfuse UI experiments](https://langfuse.com/docs/evaluation/experiments/experiments-via-ui): explicit prompt/dataset compatibility and reproducible experiment configuration. Adapt a readable run configuration summary and preflight checks for variable/reference mismatches.
- [Braintrust comparison](https://www.braintrust.dev/docs/evaluate/compare-experiments): inspecting regressions through comparison summaries, filters, and output differences. Adapt a two-run comparison with clear direction and case inspection.

These are interaction references from current official documentation. The proposed visual design is a recommendation for LLMForge, not a claim that its screens were visually compared with authenticated competitor applications. Tracing, annotation queues, multi-agent graphs, and enterprise administration are outside this refresh.

## Proposed navigation and screen structure

Primary navigation: **Overview, Prompts, Datasets, Evaluations**.

Secondary area: **Benchmarks**. Put New benchmark and Compare benchmarks inside that section. Keep existing `/experiments` routes initially to avoid unnecessary API and URL changes; use consistent product labels in the UI and documentation.

Utility navigation: **Settings, Docs**. Keep organization/project identity visible in the shell, and user account controls in one consistent location. Breadcrumbs should describe the current resource and version where relevant. Preserve project-switch confirmation and cache isolation.

### Overview

- Empty project: one prominent Create demo examples action plus Create prompt / Import dataset alternatives and a short sequence of next steps.
- Populated project: recent evaluations with status/pass counts, recent prompts with staging/production labels, and links to continue work.
- Do not claim an all-project pass rate derived from one loaded page of runs. Use existing list totals for counts where available; otherwise introduce a small aggregate endpoint only for a metric that proves useful.
- Surface a genuine outage prominently; keep healthy optional-service diagnostics out of the main workflow.

### Prompts

- Replace the two-column catalog cards with a scan-friendly table: name, latest version, staging, production, updated time. Retain search and archived filtering.
- Prompt detail tabs: **Editor & Playground, Versions, Releases, Integrate**. Put metadata/archive actions in a secondary menu or settings panel.
- Editor & Playground: clear draft/saved status, template and variables on one side, model controls and outputs on the other. Offer a wider output layout when comparing targets.
- Keep saving a version distinct from moving a release label. Show which saved version an Evaluate or Promote action applies to, including when the draft has unsaved changes.
- Start version inspection with existing snapshots. A text diff between two saved versions is a useful later enhancement, not a prerequisite for the layout refresh.
- Integrate: copyable Python SDK example using the real prompt name and chosen label/version, plus curl as an alternative. Use environment variable placeholders for credentials.

### Datasets

- Separate dataset selection from case editing. Use a selected dataset/revision in the URL so refresh and navigation retain context.
- Case table: name, input summary, reference summary; open a row editor for full values. Add/duplicate/remove rows within a draft, then explicitly save a new revision.
- Keep JSON editing and CSV/JSON imports available. Import should show validation and a preview before replacing draft cases.
- Changing between table and JSON views must preserve all supported fields and draft content. Preserve the existing behavior where formatting-only changes do not create a revision.
- Evaluate this revision should preselect the exact saved dataset revision, with a clear message if the current draft has not been saved.

### Evaluations

- Default: run history table with timestamp, prompt/version, dataset/revision, provider/model, source (demo/live/external), status, and pass/error counts.
- Setup: staged sections for prompt/version + dataset/revision, generation, and checks/judge. Keep a concise configuration summary and Start evaluation action visible.
- Detail: status/progress, explicit denominator for pass rate, processed/passed/failed/error counts, saved configuration, and a case grid. Partial runs must not look like completed results.
- Case inspection: full input, reference, output, assertion reasons, judge result, and available latency/token data. Keep summary rows compact.
- Comparison: label both runs clearly, identify which is the reference and which is the candidate, and allow filtering improved/regressed/unchanged/error cases. Existing comparison selection is sufficient initially; persistent evaluation baselines would be a separate feature.
- Preserve differing-dataset-revision restrictions and differing-assertion warnings. Do not import benchmark statistical significance claims into this comparison.
- Existing list API only supports offset/limit. Filters that claim to cover all runs need backend query support and matching totals; filtering a single page silently is unacceptable.

### Shared visual direction

Use neutral application surfaces with the existing warm accent retained for primary actions. Reduce large rounded header cards, decorative gradients, and shadow layers within the workspace. Prefer approximately 24–28px page titles, a consistent 14px control/table scale, compact toolbars, and 8–12px corner radii as initial design targets to validate visually. Keep long-form settings readable while allowing editors and result tables to use available width.

Build on the existing UI primitives rather than replacing the entire UI library. Introduce a compact page-header variant, shared form field and feedback patterns, tabs, table toolbar, status badges, and a reusable accessible detail panel. Retain responsive layouts, text status labels, visible focus, and reduced-motion behavior. Recheck existing benchmark pages when changing shared CSS.

## README restructuring

The root README should answer: what problem does this solve, what does using it look like, how do I try it, and what engineering decisions make it credible?

Recommended order:

1. **Name and one-sentence purpose.** Keep a CI badge and a small number of useful stack badges; avoid giving optional Redis the same prominence as core functionality.
2. **Product screenshot and entry links.** Add a verified deployed-app URL, docs, and a screenshot of a real comparison or prompt workbench after the redesign. Never use a fabricated screenshot as evidence of working functionality.
3. **Concrete use case.** Explain a support-ticket classifier with stable output labels/JSON references. Distinguish this live-provider example from the deterministic echo demonstration.
4. **Primary workflow.** Prompt draft → saved version → fixed dataset revision → evaluation → comparison → release → SDK. A small diagram is sufficient.
5. **Provider-free walkthrough.** Sign in/select project; create Echo demo/Greetings; evaluate v1 with exact match; save `Reply: {{query}}` as v2; evaluate on the same revision and inspect regressions; promote v1; fetch it via SDK. Explicitly state that mock output echoes the template and does not establish model quality.
6. **Capabilities.** Prompt versions/releases, versioned datasets and assertions including per-case JSON references, comparisons and cancellation, scoped workspaces/keys, Python SDK and CI gates. Add a concise secondary paragraph for reasoning benchmarks.
7. **Quick start.** Required database + Clerk configuration; mock mode as the first path; optional provider/queue/vector services separated. Keep exact commands and constraints-file instructions aligned with the working setup guides.
8. **SDK and CI example.** One complete, short example with a project key and an actually created prompt. Explain repository installation, unreleased prompts, and opt-in evaluation scope. Link the complete CLI/workflow reference.
9. **Current architecture and engineering tradeoffs.** Include Clerk/session auth, project isolation, prompt versions/labels, dataset revisions, evaluation runner/results, SDK keys, and PostgreSQL. Show optional benchmark queue/retrieval paths separately. Describe actual best-effort execution and cancellation semantics.
10. **Testing, limitations, and documentation map.** Explain what tests validate, link CI, and avoid a hardcoded test count. Keep operational limits, request-only provider credentials, and self-reported SDK submissions accurately described.

Move lengthy benchmark metrics catalogs, adaptive routing internals, built-in benchmark dataset tables, benchmark curl sequences, and design-system implementation details to focused docs, preserving links. Rename reader-facing Phase 1/2/3/4 link labels to task-oriented titles immediately; file renames can be a separate link-checked cleanup.

Specific corrections: replace the old core-question sentence and footer; rewrite the benchmark-only architecture diagram; update the landing-page route description; clarify that SDK keys are managed in Settings, not the prompt list; describe the existing organization role boundaries without implying an invitation/member-management interface that is not present.

The SDK README is substantially more aligned with the current workflow than the root README. Consolidate shared examples and terminology across it, `/docs`, and the root README instead of expanding three independent copies of the full guide.

## Phased implementation roadmap

Effort bands are relative estimates: S = a focused change, M = several related component changes, L = a screen redesign plus interaction verification. They are not elapsed-time commitments.

| Phase | Work and primary files | Dependencies / data impact | Acceptance milestone |
| --- | --- | --- | --- |
| 1 — Product language and shell (S–M) | Navigation groups, benchmark naming, complete command palette, compact header variant; `app-shell.tsx`, `ui/primitives.tsx`, `globals.css`. Draft the README outline. | No schema changes. Preserve existing routes. | A new user can identify Prompts → Datasets → Evaluations; Compare has an unambiguous scope; current routes remain reachable; displayed shortcuts work or are removed. |
| 2 — Useful overview (M) | Replace benchmark-centric overview with recent evaluations/releases and a first-run guide; dashboard and existing list clients. | Reuse current list endpoints initially. Add a project-scoped summary query only if required; no misleading page-derived aggregates. | Empty and populated projects have appropriate primary actions; the seeded demo opens exact prompt/dataset versions; API failure remains actionable. |
| 3 — Prompt workspace (M) | Compact prompt catalog, editor/playground layout, version/release/integration panels; prompts pages and components. | Existing prompt APIs and models. No version/release semantic changes. | Draft testing, Save new version, Evaluate saved version, promotion, archive/restore, and SDK instructions remain complete and clearly differentiated. |
| 4 — Evaluation inspection (L) | History table, separated setup/detail, readable summaries, case panel, regression filters; evaluations workbench/results and URL state. | Optional backend list filters must preserve tenant scoping and pagination totals. Existing run/results models remain sufficient. | v1 versus v2 demo reveals the regressed cases immediately; refresh/back retain selection; partial/cancelled/error states and comparison warnings stay accurate. |
| 5 — Dataset and settings refinement (M–L) | Case table/editor, import preview, exact revision links, grouped settings/API key guidance; dataset workbench and Settings. | Existing revision and key APIs. Dataset name search may require a small server filter. No schema migration expected. | Table/JSON round-trip loses no supported fields; formatting-only edits produce no revision; imports and unsaved-change prompts still work; Evaluate carries the selected revision. |
| 6 — README and presentation (M) | Execute the new outline; relocate benchmark details; align public docs and SDK examples; capture real screenshots; update design-system documentation. | Final screen names and validated demo behavior from prior phases. README text can begin earlier; screenshots should come last. | A fresh reader understands the use case and completes the documented demo; every example/link is valid; README, landing page, and app use consistent terminology. |

Recommended first implementation slice: Phase 1 + Phase 2, the dataset evaluation deep link, and a root README narrative correction. This produces visible alignment quickly. The larger dataset editor and output-diff enhancements can follow after the central prompt/evaluation journey is coherent.

## Validation before calling the refresh complete

Implementation status: the shell, overview, prompt catalog/workbench, evaluation history/setup/results, dataset case editor/import preview, Settings grouping, README, and related guides have been updated. Automated frontend checks are recorded in the implementation work. Browser-based visual and authenticated demo validation remain outstanding because this task's saved browser permission blocks agent access to localhost; no product screenshot has been added.

- Walk the documented demo on a fresh project: sign in, seed, evaluate v1, create v2, compare, promote v1, and fetch through the SDK. Use mock mode for repeatability.
- Test prompt drafts versus saved-version actions, failed-save recovery, import validation, cancelled/partial runs, external submissions, and incompatible comparisons.
- Verify organization/project switches, unauthorized resource links, archived items, pagination, refresh, back/forward, and deep links. URL state must never substitute for server authorization.
- Inspect actual screens at common laptop and wide desktop sizes plus a narrow mobile viewport, in both themes. Check overflow, readable contrast, sticky controls, focus order, dialog focus restoration, and reduced motion.
- Run frontend interaction tests, lint, typecheck, and production build for changed UI. Run relevant tenant-scoped backend tests if list/summary APIs change. Use the existing full CI gates before merging.
- Recheck README commands from a clean setup, Markdown links and relocated anchors, SDK examples, and screenshots against the resulting UI. Do not imply that the optional echo demonstration proves real-world model quality.

## Deferred ideas

Full observability/tracing, collaboration roles beyond existing needs, annotation queues, user-defined dashboards, persistent evaluation baselines, arbitrary multi-run comparison, independent multi-draft playgrounds, and a global full-text search service would expand scope substantially. None is necessary to make this refresh convincing. The high-value work is presenting existing capabilities as one understandable workflow.
