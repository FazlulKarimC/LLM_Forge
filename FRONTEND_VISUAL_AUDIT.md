# LLMForge frontend visual and usability audit

Audit and improvement plan only • 11 October 2026 • [Hosted application](https://llmforge.vercel.app)

## Assessment

LLMForge already has a coherent identity: charcoal surfaces, restrained warm gold actions, teal focus accents, compact workspace typography, and a shared flask mark. The public pages feel deliberate, and the workspace communicates saved versions and reproducible evaluations unusually well for a learning project. The next investment should be layout and interaction precision: important controls compete with explanatory content on narrow screens, some collection views conceal useful status, and the desktop evaluation case layout wastes half its available width.

Keep the existing fonts, palette, logo and flat surfaces. Preserve the separate Benchmarks area. More gradients, ornament or animation would not resolve the observed problems. No numerical design score is assigned.

What works well:

- Landing and docs share the logo, public header/footer, theme and clear route into the workspace. The mobile hero wraps cleanly; public controls measured about 44 px.
- Overview clearly connects prompt development to evaluations. Its metric cards and recent-run entry point stack cleanly on mobile.
- Saved prompt versions have a visible selected border. Version diffs label reference and candidate; Releases explicitly separates saving from moving production; Integrate explains pinned versions and release labels.
- Evaluation summaries distinguish failed checks from errors, show processed coverage and disclose Demo mode. Candidate/reference summaries remain clearly labelled.
- Case selection actually moves focus to the loaded detail on mobile. The inspected Unicode case remained readable; inputs, expected output and candidate/reference content were distinguishable.
- Command palette Escape restored focus to its opener. Mobile navigation Escape restored focus to Open navigation. Walkthrough arrow-key navigation changed selected tabs correctly.
- Sampled text colors have sufficient measured contrast in both themes. Small labels remain a density/readability concern, not an established contrast failure.

No application code, production records, keys, releases or evaluations were changed. The audit used existing Echo examples, read-only navigation, search filters, disclosures and temporary theme/viewport changes. No model calls, imports, saves or dispatches were submitted.

## Deployment and evidence

Local HEAD and the hosted frontend source commit matched:

`9b775fff4c5a0e62ee92e3ac1e3112c5c0d16e9a`

The docs' expanded Deployment identity also reported that commit for the backend.

| Surface | Reported build |
|---|---|
| Frontend | `dpl_HoAyeav5HHGgTPwAhNprShVRRcAc` |
| Backend API 0.1.0 | `github-37972043730-1` |

This is reported source parity, not an assertion that every runtime dependency or setting is identical. No local app was launched. All screenshots show the hosted app; source explanations refer to this matching checkout.

Viewport dimensions are CSS pixels: **desktop 1440 × 900**, **tablet 768 × 1024**, **mobile 390 × 844**. “Desktop” alone does not establish behavior at every wide size. No device, touch screen or full screen-reader session was used.

Evidence folder: [sanitized screenshots](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/>). Forty JPEG screenshots were retained after visual review. Workspace account/project headers were excluded; saved content is limited to existing audit examples. Public-page captures retain branding. Crops sometimes show content below the fold; they are not all viewport-sized. Images document current behavior, not proposed mockups.

**Evidence labels:** H = hosted visual/DOM observation; S = source confirmation; S-only = source concern without a reproduced hosted failure. Priority is relative user impact: P1 first, P2 next, P3 optional. Effort is a rough implementation size, not a delivery commitment: S = a few hours, M = roughly 1–2 days, L = several days including responsive verification.

## Coverage matrix

D/L mean dark/light. A listed combination describes inspected coverage, not an exhaustive test of every control or state.

| Route / surface | Hosted states inspected | Themes / viewports | Evidence / limits |
|---|---|---|---|
| `/` landing | Hero, workflow content, three walkthrough tabs, mobile public menu | D/L desktop; D/L mobile; D tablet | [Desktop light](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/landing-desktop-light.jpg>), [mobile light](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/landing-mobile-light.jpg>). Walkthrough arrow keys and menu Escape inspected. |
| Product walkthrough | Benchmarks, Prompts, Evaluations; read-only example images | D desktop/mobile; L landing | [Prompt walkthrough](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/walkthrough-prompts-desktop-dark.jpg>), [evaluation walkthrough](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/walkthrough-evaluations-mobile-dark.jpg>). Illustrations are not evidence of live benchmark execution. |
| `/docs` | Guide sections, SDK/code examples, troubleshooting, provider loading/loaded, expanded deployment identity | L desktop/mobile; D tablet/mobile | [Desktop](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/docs-desktop-light.jpg>), [tablet](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/docs-tablet-dark.jpg>), [mobile](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/docs-mobile-light.jpg>). Long code execution/copy behavior not tested. |
| `/sign-in` and recovery | Valid session redirected to workspace; brief session loading | L desktop | Signed-out Clerk form and expired-session recovery **unverified**. Shared AuthFrame/WorkspaceProvider source inspected. No sign-out or credential manipulation. |
| `/dashboard` | Populated overview: 5 prompts, 3 datasets, 33 runs | D/L desktop; L tablet/mobile | [Desktop dark](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/overview-desktop-dark.jpg>), [mobile light](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/overview-mobile-light.jpg>). Create demo examples not clicked. |
| Application shell | Expanded/collapsed sidebar, theme switch, command palette, mobile drawer | D/L desktop; L mobile drawer | [Collapsed sidebar](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/shell-collapsed-light.jpg>), [palette](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/command-palette-light.jpg>). Escape/focus return inspected; full tab-cycle trap and screen-reader output not tested. |
| `/prompts` | Populated catalog, long names and release columns | D desktop/mobile | [Mobile catalog](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/prompts-catalog-mobile-dark.jpg>). Archived/zero-record/server-error states unverified. |
| `/prompts/new` | Initial editor draft, disabled Create, blank sample input/output | D mobile | Read-only form inspection; no typing, save, preview or Run. Chat-message editor state not visually verified. |
| `/prompts/[id]` | Saved Echo v2 editor/playground; Versions, Releases, Integrate; metadata/archive section; promotion confirmation opened then cancelled | D desktop/mobile; L tablet/mobile editor | [Editor desktop](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/prompt-editor-desktop-dark.jpg>), [tablet](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/prompt-editor-tablet-light.jpg>), [Versions](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/prompt-versions-desktop-dark.jpg>), [Releases](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/prompt-releases-desktop-dark.jpg>), [Integrate](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/prompt-integrate-desktop-dark.jpg>). No promotion confirmed; provider results, dirty-draft navigation dialog and successful saves unverified. |
| `/datasets` | Populated catalog; existing Echo revision 2, four cases; selection, table, import disclosure with blank content | L desktop/mobile; D desktop import | [Desktop editor](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/dataset-editor-desktop-light.jpg>), [mobile](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/dataset-editor-mobile-light.jpg>), [import disclosure](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/dataset-import-desktop-dark.jpg>). Import preview/apply/error and edited JSON states unverified; no file uploaded. |
| `/evaluations?view=new` | Initial setup, available provider help, disabled action, summary, selectors | L desktop/mobile | [Desktop setup](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-setup-desktop-light.jpg>), [mobile setup](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-setup-mobile-light.jpg>). No evaluation started. Judge/credential validation states unverified. |
| `/evaluations` history | 33 runs; completed/failed/cancelled/external/saved-output rows; no-match search | D/L desktop; D mobile | [Desktop history](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-history-desktop-light.jpg>), [mobile history](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-history-mobile-dark.jpg>), [no-match state](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-history-empty-search-dark.jpg>). Genuine first-use empty state not reproduced. |
| Evaluation result / comparison / selected case | Existing completed Echo candidate 0% vs reference 100%; four cases; Unicode detail; existing failed runner with 2/4 processed | L desktop comparison; D tablet/mobile results; D desktop/mobile failure | [Comparison](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-comparison-desktop-light.jpg>), [desktop case](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-case-comparison-desktop-light.jpg>), [mobile case](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-case-mobile-dark.jpg>), [failed run](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-failed-desktop-dark.jpg>). Failed runner is not a reproduced resumable interrupted state. |
| Evaluators (within Evaluations) | Populated catalog; existing exact-match definition/editor and sample-test form | L desktop/mobile | [Editor](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluator-editor-desktop-light.jpg>). No Test/save; judge rubric, validation and new-version success unverified. |
| `/experiments` Benchmarks | Zero-record catalog with filters and empty message | L desktop | [Empty catalog](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/benchmarks-catalog-empty-light.jpg>). No safe populated benchmark records in the inspected project. |
| `/experiments/new` | Blank setup, defaults, routing/regression/lineage sections, presets, draft/run actions | L desktop/tablet/mobile | [Desktop](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/benchmark-setup-desktop-light.jpg>), [tablet](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/benchmark-setup-tablet-light.jpg>), [mobile](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/benchmark-setup-mobile-light.jpg>). No create/run. Dark setup not visually verified. |
| `/experiments/[id]`, `/experiments/compare` | Comparison zero-completed state and Create action | L desktop/mobile comparison | [Empty comparison](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/benchmark-comparison-empty-light.jpg>). Populated detail/comparison/profile/routing charts **S-only, unverified visually**. |
| `/settings` keys | Existing project page, capability explanations, blank creation controls | L desktop | [Cropped key form](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/settings-key-form-desktop-light.jpg>). Key inventory/account areas excluded. No create/reveal/revoke; mobile and successful key creation unverified. |
| Feedback / overlays | Loading session/providers/prompts/datasets; failed-run message; no-match history; palette/drawer; inline release confirmation | Various combinations above | Notifications region present. Active success/error toasts, provider failures, unsaved-change dialog and recoverable interruption need safe fixtures. |

## Five highest-impact improvements

| Order | Finding | Why it matters | Benefiting screens |
|---|---|---|---|
| 1 | FVA-001: shrink narrow evaluation summary | The sticky summary consumes about 331 px before the first prompt selector; configuring a run becomes a scrolling exercise. | Evaluation setup on mobile/tablet |
| 2 | FVA-003: repair desktop case pairing | A selected detail occupies only about 605 px of a 1134 px result area while the table sits elsewhere. Users cannot inspect rows and evidence efficiently. | Evaluation results and comparison |
| 3 | FVA-002: bring playground within reach | A 340 px editor minimum plus metadata pushes Run prompt to document y ≈1341 on mobile. | Saved/new prompt editor, playground |
| 4 | FVA-004: make dataset selection explicit | Catalog rows look identical after selection; the detail header states only revision/case count. | Dataset catalog/editor, evaluation handoff |
| 5 | FVA-005: adapt collections to narrow screens | History status/progress and prompt release information sit off-screen inside wide tables. | Evaluation history, prompt catalog, dataset cases |

### FVA-001 — Narrow setup summary overwhelms the form

**P1 · H + S · M.** Location: evaluation setup at 390 × 844, light.

**Observed:** Run configuration measured about 331 px high, above the configuration fields. The first Prompt selector began around document y=911. At widths below 1100 px the summary becomes sticky at top=56 px, retaining paragraphs and nested framing. Start is visible in that summary; the problem is its footprint, not a missing action.

**Evidence:** [Mobile](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-setup-mobile-light.jpg>) / [desktop](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-setup-desktop-light.jpg>).

**Impact/recommendation:** Collapse explanatory limits/settings into a disclosure on narrow screens. Keep a compact 96–112 px summary with selected versions, call count and Start; use a 12 px gap and 16 px padding. Place required selections before extended provider help. Preserve the useful 280 px desktop side summary.

**Acceptance:** At 390 and 768 px the first required selector appears within the initial viewport; the collapsed sticky summary stays within the proposed height budget without covering focused fields. Full limits and effective settings remain keyboard-accessible. Changing selections updates the summary, and dispatch stays explicit.

**Source:** [summary component](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/features/evaluations/components/evaluations-workbench.tsx:557>), [narrow sticky rules](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/app/globals.css:1153>). Tablet setup itself was not separately captured; verify it during implementation.

### FVA-002 — Prompt testing is too far below the editor on mobile

**P1 · H + S · M.** Location: saved Echo editor/playground; desktop pairing becomes stacked at 768 px.

**Observed:** The textarea retains a 340 px minimum for a one-line template. At 390 px Run prompt started near document y=1341, following editor help, configuration, notes and integrity controls.

**Evidence:** [Desktop](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/prompt-editor-desktop-dark.jpg>), [mobile](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/prompt-editor-mobile-dark.jpg>), [tablet](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/prompt-editor-tablet-light.jpg>).

| Desktop: editor and playground share the workspace | Mobile: editor consumes the initial view |
|---|---|
| ![Desktop saved Echo editor](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/prompt-editor-desktop-dark.jpg>) | ![Mobile saved Echo editor](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/prompt-editor-mobile-dark.jpg>) |

Both images show current hosted behavior, with account/project headers excluded. The mobile image ends before the playground; measured document position establishes the distance to Run.

**Impact/recommendation:** Keep the desktop editor/playground pair. Use an approximately 220 px initial editor height on mobile, 280 px on tablet, with manual expansion. Add a visible Editor / Playground switch or jump action on narrow screens; keep optional metadata collapsed. Preserve draft and sample inputs when switching.

**Acceptance:** Run and inputs are reachable from a visible first-screen affordance at 390/768 px. A short template does not reserve 340 px of blank space. Long templates remain editable, and no model call happens from navigation or preview.

**Source:** [textarea minimum](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/app/globals.css:993>), [prompt workbench](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/features/prompts/components/prompt-workbench.tsx:241>), [playground](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/features/prompts/components/prompt-playground.tsx>).

### FVA-003 — Desktop case detail uses only one column of a two-column grid

**P1 · H + S · M.** Location: paginated evaluation result/comparison with a selected case.

**Observed:** At 1440 px the selected case panel measured about 604.5 px within a 1134.4 px results area, leaving a large empty right side. The case table renders above it. Source confirms PagedResults renders the table outside ResultsGrid and passes hideTable; the remaining detail becomes the lone child of the desktop two-column case-inspection grid.

**Evidence:** [Selected comparison case and unused space](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-case-comparison-desktop-light.jpg>). [Mobile detail](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-case-mobile-dark.jpg>) shows the stacked state and working focus transfer.

**Impact/recommendation:** Put the paged table and detail in one shared inspection layout at ≥1100 px, with roughly 560 px table and 340 px detail minimums. Keep scores/coverage above this pair. Below the breakpoint, stack and retain focus transfer. A minimal initial repair is making a lone detail span both columns; the full paired layout is the intended improvement.

**Acceptance:** At 1440 px no accidental empty grid column remains; selecting a row displays its evidence beside the table. At 390/768 px content stacks without page overflow. Candidate/reference labels, compatibility warnings, lazy detail fetching, pagination and URL-selected case/filter survive.

**Source:** [PagedResults](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/features/evaluations/components/paged-results.tsx:29>), [ResultsGrid](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/features/evaluations/components/evaluation-results.tsx:278>), [desktop grid](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/app/globals.css:1087>).

### FVA-004 — Dataset selection lacks a persistent visual identity

**P2 · H + S · S for state; M for mobile arrangement.**

**Observed:** Selecting Echo revision 2 leaves catalog buttons visually equivalent; no selected class/aria-pressed was present. The editor header says “Revision 2 · 4 cases” and the identity is only in the editable Dataset name field. On mobile the catalog precedes the whole editor.

**Evidence:** [Desktop selection](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/dataset-editor-desktop-light.jpg>) / [mobile stack](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/dataset-editor-mobile-light.jpg>).

**Impact/recommendation:** Reuse the selected border/tint treatment from prompt Versions, plus aria-pressed on the active catalog button. Head the editor with the saved dataset name; put revision and case count on a secondary line. On mobile provide a collapsible dataset chooser above the editor.

**Acceptance:** Visual and accessible selected state agree. Selecting another dataset updates heading and revision. At 390 px a chosen dataset's identity and editor entry point remain visible without traversing the full catalog. Dirty-draft protections remain intact.

**Source:** [catalog](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/features/datasets/components/datasets-workbench.tsx:263>), [revision heading](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/features/datasets/components/datasets-workbench.tsx:326>).

### FVA-005 — Wide collection tables conceal important mobile information

**P2 · H + S · L across three collections, M for history first.**

**Observed:** History's 860 px table sits in about 325 px of available width on mobile; prompt catalog is at least 680 px wide. Dataset cases measured about 616 px inside a 308 px container. Overflow is contained: this was not whole-page horizontal overflow. Users initially see names while status, progress, staging/production or actions are farther right. The inspected history scroll container lacked tabindex.

**Evidence:** [History](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-history-mobile-dark.jpg>), [prompts](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/prompts-catalog-mobile-dark.jpg>), [datasets](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/dataset-editor-mobile-light.jpg>).

**Impact/recommendation:** Use shared compact collection rows below 640 px: left-aligned name, visible status/version, progress and one clear detail action; reveal secondary metadata on expansion. Keep full tables on wider screens. For retained scrollable tables, give an obvious overflow cue and a keyboard-reachable, labelled scroll region. Use tabular numerals and consistent right alignment for numeric columns.

**Acceptance:** At 390 px history status/progress and prompt production version are visible without horizontal scrolling. Long names wrap to two lines or truncate with an accessible full name. Keyboard users can access every retained column; no page-level overflow at 390/768/1440 px. Do not replace semantic tables with unlabeled div grids.

**Source:** [history table](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/features/evaluations/components/evaluations-workbench.tsx:724>), [prompt catalog table](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/app/(app)/prompts/page.tsx:123>), [dataset workbench](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/features/datasets/components/datasets-workbench.tsx:545>).

### FVA-006 — A no-match search is presented as a first-use empty workspace

**P2 · H + S · S.**

**Observed:** Searching the populated 33-run history for an unmatched name displays “No runs yet” and suggests creating a prompt/dataset. The unfiltered zero-record benchmark catalog instead says “No benchmarks match this slice.” Both blur distinct causes.

**Evidence:** [History no-match](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-history-empty-search-dark.jpg>), [benchmark zero-record catalog](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/benchmarks-catalog-empty-light.jpg>).

**Impact/recommendation:** Give collection components distinct loading, first-use, filtered-empty and error variants. Filtered-empty should explain the active search/status and offer Clear filters. Genuine empty should explain the workflow and link to creation. Errors need a retry action rather than a zero-record message.

**Acceptance:** Searching existing runs never claims there are no runs at all. Clearing filters restores the populated table. Empty benchmark copy fits an unfiltered new workspace. Loading/error variants do not flash first-use text.

**Source:** [history empty branch](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/features/evaluations/components/evaluations-workbench.tsx:718>), [benchmark empty branch](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/app/(app)/experiments/page.tsx:277>). Transport-error transitions were not reproduced.

### FVA-007 — Benchmark screens change the user's vocabulary mid-flow

**P2 · H + S · S.**

**Observed:** Navigation says Benchmarks, but setup says “Experiment builder,” “Back to experiments” and “next evaluation run.” Comparison says “Compare experiments side by side.” This weakens the deliberate distinction from Evaluations.

**Evidence:** [Setup](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/benchmark-setup-desktop-light.jpg>), [comparison](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/benchmark-comparison-empty-light.jpg>).

**Recommendation:** Use Benchmark consistently in visible titles, back links, CTAs and help. Explain once that a benchmark compares model/reasoning configurations. Internal /experiments routes and API names can stay as they are.

**Acceptance:** Overview → Benchmarks → setup → detail → comparison uses consistent product terminology. Prompt Evaluations still describes fixed prompt/dataset snapshots; docs preserve the distinction.

**Source:** [setup copy](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/app/(app)/experiments/new/page.tsx:364>), [comparison](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/app/(app)/experiments/compare/page.tsx>), [detail copy](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/app/(app)/experiments/[id]/page.tsx:149>). Detail recommendations derive from source; populated detail was not visually inspected.

### FVA-008 — Control density and nested surface shapes drift between workspaces

**P2 · H + S · M shared cleanup.**

**Observed:** Prompt/evaluation controls measured roughly 35 px high while benchmark inputs were about 43.7 px despite a 34 px workspace minimum. Routing controls sit in a rounded nested card; the command palette uses a 26 px outer radius and 16 px rows, contrasting with 7 px controls/10 px panels elsewhere.

**Evidence:** [Benchmark tablet](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/benchmark-setup-tablet-light.jpg>), [evaluation setup](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-setup-desktop-light.jpg>), [palette](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/command-palette-light.jpg>).

**Impact/recommendation:** Introduce explicit compact/public control variants rather than mixing minimum height with generous page-specific padding. Use 34–36 px desktop workspace and 40 px narrow controls, 44–45 px public controls; radius 7 px controls, 10 px panels, a documented 12–16 px overlay exception. Use separators instead of redundant inner cards where grouping remains clear.

**Acceptance:** Equivalent input/select/button variants share actual rendered heights, including padding. Palette, routing, prompt and evaluation controls use documented radii. Both themes retain readable focus/disabled states. Public narrative sizing can remain larger.

**Source:** [density overrides](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/app/globals.css:899>), [benchmark sections](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/components/experiments/new-experiment-sections.tsx:33>), [palette](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/components/app-shell.tsx:240>). This is a consistency/readability recommendation, not a measured accessibility violation.

### FVA-009 — Benchmark draft/run choices arrive after a very long mobile form

**P2 · H + S · M.**

**Observed:** At 390 px the unfilled Create and run button begins around document y=3948. Routing, execution, regression and lineage settings occupy the initial scroll path; side presets become another stacked section.

**Evidence:** [Mobile setup](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/benchmark-setup-mobile-light.jpg>), [desktop setup](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/benchmark-setup-desktop-light.jpg>).

**Recommendation:** Group required name/model/dataset/reasoning fields first; move advanced routing/regression/lineage into labelled disclosures. Add a compact summary/action bar with clearly separate **Save draft** and **Create and run** actions, with their current enabled/validation state. Do not make draft save trigger execution.

**Acceptance:** At 390/768 px the draft/run choices are discoverable without scrolling through optional sections. Expanded controls remain labelled and keyboard-operable. Provider credential requirements and execution implications remain visible before dispatch.

**Source:** [sections and actions](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/app/(app)/experiments/new/page.tsx:394>). No filled/validated submission state was inspected.

### FVA-010 — DESIGN_SYSTEM.md contains contradictory prescriptions

**P2 · S-only documentation finding · S.**

**Observed in source:** The document gives 232 px and 264 px sidebar widths; describes input-shell as 14 px radius / 2.9rem while current workspace rules use 7 px / 34 px minimum; says light accent/status colors are unchanged despite darker light overrides; prohibits lifting/glow while a later animation table prescribes lift, radial backgrounds and dot-grid treatment.

**Evidence:** No screenshot can establish a documentation contradiction. [Current flat workspace](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/overview-desktop-dark.jpg>) is visual context only.

**Impact/recommendation:** Consolidate one canonical token/density table with separate public/workspace variants. Remove stale duplicate tables and motion/background prescriptions. Record the existing theme-specific semantic colors. State deliberate exceptions rather than relying on later overrides to win.

**Acceptance:** One searchable specification per sidebar width, radius, control height and motion rule; values agree with globals.css and measured rendered controls. No conflicting legacy animation guidance remains.

**Source:** [theme table](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/DESIGN_SYSTEM.md:47>), [input table](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/DESIGN_SYSTEM.md:150>), [legacy sidebar](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/DESIGN_SYSTEM.md:392>), [animation table](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/DESIGN_SYSTEM.md:414>), [current workspace table](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/DESIGN_SYSTEM.md:430>), [light overrides](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/app/globals.css:1180>).

### FVA-011 — Collapsed Repository link has no accessible name

**P2 · H DOM + S · S.**

**Observed:** With sidebar collapsed, the DOM accessibility snapshot exposes an unnamed link for the GitHub icon. The normal navigation links retain title-derived names; the Repository anchor does not.

**Evidence:** [Collapsed rail](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/shell-collapsed-light.jpg>) shows the icon-only context. The absent name is a DOM observation, not visible in pixels.

**Recommendation:** Give this anchor a stable aria-label="Repository" and optional tooltip/title; mark its decorative icon aria-hidden. Keep the expanded visible label.

**Acceptance:** Accessibility tree names the link Repository in expanded/collapsed modes. Tab focus is visible; Enter opens the expected repository. Verify with a screen reader during implementation.

**Source:** [conditional Repository label](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/components/app-shell.tsx:419>). No broader claim that all collapsed links are unnamed.

### FVA-012 — Reduced-motion support has an uncovered JavaScript path

**P2 · S-only concern · S/M.**

**Source observation:** globals.css reduces CSS animation/transition duration, but AnimatedNumber schedules a 700 ms requestAnimationFrame interpolation without checking reduced motion. Palette/drawer use Framer Motion transforms without an explicit preference branch in the inspected shell.

**Evidence:** [Palette context](C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/command-palette-light.jpg); it does not demonstrate reduced-motion behavior. Browser preference emulation was not available in this audit.

**Recommendation:** Use a shared reduced-motion preference hook or Framer's existing hook. Render final numbers immediately when requested; remove overlay movement while keeping correct focus timing.

**Acceptance:** In a controlled prefers-reduced-motion session counters show their terminal value without interpolation and overlays do not scale/slide. Normal mode remains restrained; closing still restores focus.

**Source:** [AnimatedNumber](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/components/ui/primitives.tsx:201>), [palette motion](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/components/app-shell.tsx:224>), [CSS preference rules](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/app/globals.css:1131>). Treat as a verification task before calling it a hosted failure.

### FVA-013 — Release actions do not distinguish already-assigned versions

**P3 · H + S · S.**

**Observed:** Saved v2 is already assigned to staging, yet that row offers enabled Promote v2 just like production, which still points to v1. The confirmation copy correctly explains the next fetch.

**Evidence:** [Release rows](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/prompt-releases-desktop-dark.jpg>).

**Recommendation:** Show Assigned v2 as passive state, or disable the same-version promotion with clear text. Keep Promote v2 on rows that would change. Retain the explicit saved-snapshot confirmation for actual changes.

**Acceptance:** Staging already at v2 requires no redundant action; production at v1 clearly offers the actual transition. Selecting v1 updates those states. Custom labels and removals remain separately named.

**Source:** [release actions](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/features/prompts/components/prompt-workbench.tsx:565>). No release was changed.

### FVA-014 — Single-page result tables retain dead pagination controls

**P3 · H + S · S.**

**Observed:** Four matching cases show Previous and Next, both disabled, despite only one page. This adds an unnecessary toolbar and is contrary to the document's collection guidance.

**Evidence:** [Tablet result](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-result-tablet-dark.jpg>), [failed result](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/evaluation-failed-desktop-dark.jpg>).

**Recommendation:** Share a pagination footer that keeps the matching count but hides navigation when total fits one page. Keep loading semantics and boundary disabling for multipage results.

**Acceptance:** 0–25 cases have no Previous/Next buttons; 26+ cases retain correctly labelled navigation. Filter changes reset page appropriately without losing the selected run.

**Source:** [always-rendered pager](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/frontend/src/features/evaluations/components/paged-results.tsx:32>), [collection guidance](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/DESIGN_SYSTEM.md:433>).

## Measured colors and sizing

Contrast was calculated from computed CSS token values, not screenshot appearance. Raw colors are in [theme-tokens.json](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/theme-tokens.json>); calculations/method are in [contrast-measurements.json](<C:/Users/FAZLUL/Desktop/MainProject/LlmForge/artifacts/frontend-visual-audit/contrast-measurements.json>). CSS Lab D50 values were converted through Bradford D65 adaptation to linear sRGB, with out-of-gamut channels clipped. Ratios below are rounded, opaque token pairs only.

| Text/background pair | Dark | Light |
|---|---:|---:|
| Foreground / background | 16.60 | 16.17 |
| Muted foreground / surface 1 | 7.31 | 5.66 |
| Muted foreground / surface 2 | 6.73 | 5.25 |
| Primary foreground / primary button | 10.00 | 10.00 |
| Primary text / surface 1 | 9.68 | 7.06 |
| Accent / surface 1 | 8.45 | 6.44 |
| Success / surface 1 | 8.85 | 6.36 |
| Destructive / surface 1 | 5.79 | 5.95 |
| Warning / surface 1 | 10.73 | 5.96 |

All sampled pairs exceed the normal-text 4.5:1 criterion described by [W3C Contrast Minimum](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html). This is **not a full WCAG conformance audit**. Alpha-composited status pills, focus outlines, borders, disabled controls and every contextual text pair were not measured. Small 11.2 px table headers and 11.52 px help remain readability preferences to verify at zoom, not evidence of contrast failure.

Suggested shared values follow the existing compact workbench direction:

| Token / rule | Observed / existing direction | Proposed use |
|---|---|---|
| Sidebar / header | 232 px / about 56.8 px workspace header | Keep; document once. Public header can remain larger. |
| Page title | About 25.6 px desktop, 22.4 px mobile | Keep current hierarchy; no new font family. |
| Section / body / helper | Existing 16 px section, roughly 13 px controls, smaller helpers | 16 px section; 13–14 px body/controls; 12–13 px meaningful helper. Review dense 11 px metadata before increasing globally. |
| Workspace controls | 34 px minimum; rendered variants vary | 34–36 px desktop, 40 px narrow including inputs/selects; public ≥44 px. This is a product sizing target, not a claim that every smaller control fails WCAG. |
| Shape | 7 px controls, 10 px panels | Keep; document 12–16 px overlay exception. |
| Spacing | Compact panels and page gaps | Reuse 4/8/12/16/24/32 px steps; 16 px compact panel padding, 24 px major section gap, 12 px related-control gap. |
| Editor | 340 px minimum across layouts | 340 px desktop, about 280 px tablet, 220 px mobile initial; expand when needed. |
| Narrow setup summary | About 331 px | Collapsed 96–112 px budget; expand help on request. |

## Phased implementation plan

### Phase 1 — Shared tokens and components

First reconcile DESIGN_SYSTEM.md with globals.css (FVA-010), then implement actual rendered control-height variants and the radius scale (FVA-008). Extract small collection empty-state and pagination helpers (FVA-006/014). Add the Repository accessible name (FVA-011). Keep this change focused: no API work, no framework migration, no new styling library.

**Quick wins:** clearer no-match text/Clear filters; correct selected dataset class/aria-pressed; Repository label; consistent visible Benchmark vocabulary; already-assigned release state; hide one-page navigation. Each can be reviewed independently.

**Exit check:** desktop/light/dark screenshots of representative controls and catalog states agree with the canonical spec. Required semantics remain unchanged.

### Phase 2 — Core workspace layout and hierarchy

Repair the paginated table/detail composition first (FVA-003). Then compact narrow evaluation configuration (FVA-001), prompt editor/playground access (FVA-002), and dataset chooser/header (FVA-004). Start responsive collection rows with evaluation history before applying them to prompts/datasets (FVA-005).

**Larger changes:** paired case inspection, narrow summary/disclosure, editor/playground switching and responsive collections. Prefer small shared components with existing queries, state and routes.

**Exit check:** compare 1440/768/390 px views of one saved Echo prompt, one dataset and the existing candidate/reference pair. Keep pinned versions, draft state, comparison compatibility, processed coverage and case focus behavior.

### Phase 3 — Interaction, responsive and accessibility polish

Add JavaScript reduced-motion handling and verify it in a controlled browser session (FVA-012). Verify drawer/palette full Tab/Shift+Tab cycling, visible focus, labelled scroll regions, zoom and long-name wrapping. Improve benchmark mobile grouping/action visibility (FVA-009). Check saving/loading/validation and notifications using disposable local fixtures, not hosted production mutations.

**Exit check:** keyboard-only journeys, both themes, reduced motion, 200% zoom, long/multiline/Unicode values and asynchronous detail loading. Recheck sampled text plus composited focus/status colors after styling changes.

### Phase 4 — Optional refinements

Polish already-assigned releases and one-page footers if not taken as quick wins. Review redundant headings in Settings and nonessential nested frames after core improvements. Inspect populated benchmark charts and authentication/recovery before proposing changes to them. Keep landing/walkthrough storytelling and documentation readable; do not add decoration solely to make them seem premium.

## Remaining verification and safe fixtures

- **Auth/session:** use a separate disposable browser session or local auth fixture for signed-out Clerk, expired/401 refresh, offline recovery and return-path preservation. Do not sign the current user out to manufacture these states.
- **Benchmark detail/comparison:** two existing safe completed benchmark fixtures with contrasting metrics, long names, missing values and partial/interrupted results. Create them in a local/disposable environment; no hosted model dispatch is needed.
- **Forms/feedback:** local mocked responses for successful save, validation, import preview/apply, provider failure, key-created confirmation without a real secret, and success/error notifications.
- **Evaluation interruption:** a fixture representing an actually resumable interrupted run, distinct from an ordinary failed runner. Verify fresh-credential messaging and checkpoint coverage without executing a provider.
- **Accessibility:** real preference emulation, screen reader, full focus-trap loop, touch input and zoom tests. This audit's DOM and contrast sampling does not substitute for those checks.

Backend deadlock logs supplied earlier explain a separate operational investigation. A visual audit cannot establish the failed run's database cause or fix it through presentation. Preserve clear failure messaging while investigating that issue separately.

The deliverable is ready for review. Implementation, commit, merge, push and deployment remain outside this request.
