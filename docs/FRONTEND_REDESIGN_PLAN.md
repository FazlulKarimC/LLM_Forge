# LLMForge frontend redesign plan

Prepared 2026-10-01 after reviewing Langfuse public frontend sources and screenshots, and inspecting the authenticated local LLMForge workspace.

## Objective and evidence

Make prompt development, dataset editing, and evaluation inspection feel like one focused workspace. Apply Langfuse's principles of consistent page structure, compact controls, contextual actions, and adjacent data/detail views while keeping LLMForge's name, dark/light themes, gold primary actions, and teal accents.

Rendered screens inspected: Overview, new prompt/editor/playground, dataset empty state and unsaved case editor, evaluation setup and empty history, and prompt list shell. The default project had no saved records. Populated versions, releases, run results, and comparisons are source-reviewed; their visual layout must be validated with demo data before implementation decisions are finalized.

Existing improvements are the starting point: grouped navigation, prompt catalog table, prompt detail tabs, dataset case table/import preview, evaluation history/setup separation, and regression filters. This work refines their placement and interaction rather than rebuilding those capabilities.

## Placement principles

1. **Stable hierarchy:** workspace context → page identity/actions → tabs or filters → working content → contextual detail.
2. **Actions beside their subject:** Save beside draft state; Run beside execution settings/output; revision selection beside dataset identity; case filters beside the case table.
3. **Data receives the space:** explanatory copy, metadata, and advanced settings should not push the template or results below the first screen.
4. **Progressive disclosure:** keep normal controls visible; put uncommon parameters, integration instructions, and metadata in appropriate secondary areas.
5. **Consistent density:** shared controls, headers, tables, feedback, and detail panels use the same spacing and typography.
6. **Responsive continuity:** desktop split views become stacked views or focused drawers on smaller screens, preserving selection and actions.

## Target shell and page templates

Desktop structure:

```text
┌───────────────────┬─────────────────────────────────────────────────────┐
│ Brand/workspace   │ Workspace context                         Account   │
│                   ├─────────────────────────────────────────────────────┤
│ Overview          │ Breadcrumb / page title              Page actions  │
│ Prompts           │ Tabs or search/filter toolbar                       │
│ Datasets          ├─────────────────────────────────────────────────────┤
│ Evaluations       │                                                     │
│                   │ Primary working content          Contextual detail │
│ Benchmarks        │                                                     │
│                   │                                                     │
│ Settings / Docs   │                                                     │
│ Theme / commands  │                                                     │
└───────────────────┴─────────────────────────────────────────────────────┘
```

- Use a narrower sidebar, initially around 220–240px, with 34–38px navigation rows. Keep existing group semantics; move utilities toward the bottom. Replace the large command-palette card with one compact command entry.
- Keep organization/project context together and the account control at the opposite end of a compact global bar. Remove repeated Home/GitHub/command affordances from the main work area; retain their destinations in utility navigation.
- Replace the full-width eyebrow strip with a short breadcrumb or small contextual label. Use a compact title/action row, followed immediately by tabs or filters. Make the page action area sticky where tasks are long; avoid multiple overlapping sticky rows.
- Provide three shared page patterns: **collection** (toolbar + table), **workbench** (editor/table + adjacent detail), and **settings** (readable bounded column).
- Initial visual targets: 24–28px page titles, 13–14px table/control text, 32–36px standard controls, 6–10px control radii, 10–12px panel radii, and 16–20px working-area padding. Validate actual readability and touch targets; these are starting values, not rigid requirements.
- Flat neutral surfaces and subtle separators carry hierarchy. Reserve gold for primary actions, teal for focus/links, and semantic colors for status with text labels.

## Screen-by-screen placement

| Screen | Proposed arrangement | Behavior to preserve |
| --- | --- | --- |
| Overview | Compact page header; concise first-run panel for an empty project; small inventory strip; recent evaluations and prompts below. Established projects get recent work before onboarding. Service status stays secondary unless unhealthy. | Accurate project totals, actionable failures, deterministic demo links. |
| Prompt catalog | Title + New prompt; one compact search/archive toolbar; full-width prompt table; footer pagination only when useful. | Server search, archive selection, release labels, pagination scope. |
| Prompt creation | Compact identity fields above a large template editor; playground alongside it on desktop; creation action in the page toolbar. | Required name/template fields, template format, draft testing. |
| Saved prompt | Title/version/draft-state + Save/Evaluate toolbar; quiet tabs directly below; large template editor left, playground right. Name/description edits move to Settings or a metadata popover. Version notes appear beside Save. | Save version differs from updating metadata and moving release labels; Evaluate uses an explicit saved version; archive/restore and unsaved changes remain reliable. |
| Playground | Variables grouped close to the template; compact provider/model toolbar above each output; advanced generation parameters expandable; persistent Run controls. For comparisons, use a wider output area with aligned model columns rather than vertically stacking all model forms. | Existing single-draft, up-to-three-model execution; request-only keys, shared generation settings, preview, per-model failures/usage. Independent prompt drafts are outside this redesign. |
| Datasets | Narrow dataset navigator; selected dataset name/revision + Save/Evaluate toolbar; cases take the main area; selecting a case opens an adjacent editor on wide screens or accessible drawer on smaller screens. Put description in metadata and import/JSON in secondary controls. | Exact revision links, draft/revision distinction, all supported fields, table/JSON round-trip, import validation/preview, formatting-only changes. |
| Evaluation history | Compact header + New evaluation; run table with prompt/version, dataset/revision, model/source, progress/status; pagination below. Add only filters supported by the list API. | Correct totals, deep links and selection across refresh/back, tenant scoping. |
| Evaluation setup | Three clearly labeled groups: Prompt & dataset, Generation, Checks & judge. Keep a concise configuration summary and Start action in a sticky right column on desktop; use an accessible action footer/summary on narrow screens. Replace always-visible selector pagination with compact controls shown only when more pages exist. | Exact versions, variable/reference checks, optional judge, request-only credentials, execution limits and error recovery. |
| Run detail | Compact run identity/status and metric strip; actions/filter bar; case table with selected-case detail adjacent. Full input/reference/output/check reasons belong in detail rather than tall table rows. | Pass-rate denominator, incomplete/cancelled states, case errors, export, cancellation and source labels. |
| Comparison | Explicit Reference and Candidate headers; aligned run summaries; improved/regressed/unchanged/error filters beside case table; selected case shows reference/candidate outputs side by side, with checks and reasons directly below. | Different-revision restriction and different-assertion warning; unmatched and unfinished cases; no benchmark significance claims. Text diff is optional after the basic placement works. |
| Settings | Small section navigation and bounded form groups for project/access, API keys, workspace/account. Keep destructive controls secondary and clearly labeled. | Existing owner permissions, key scopes, one-time secret display and workspace isolation. |
| Benchmarks | Adopt the same compact shell, toolbar, table, and inspection patterns while retaining separate benchmark terminology and statistical comparison. | Benchmark provenance, routing controls, diagnostics, and statistical meaning. |

## Implementation order and review milestones

### 1. Verify populated workflows

Use a dedicated design-review project with the provider-free Echo demo. Capture the current populated prompt, dataset, run detail, and comparison screens. Confirm demo evaluation requests explicitly use mock generation; avoid relying on the backend's default inference engine. Record desktop, laptop, and narrow-screen behavior.

**Done when:** baseline screenshots and the v1/v2 regression journey are verified, and any additional placement issues are incorporated into this plan.

### 2. Establish shared components and compact shell

Update `app-shell.tsx`, `ui/primitives.tsx`, shared controls and `globals.css`. Introduce reusable page layout/header, tabs, table toolbar/footer, form field/group, feedback, status badge, action bar, and detail panel patterns. Consolidate the duplicated classes in `prompts/prompt-ui.tsx` and Settings. Scope workspace styling so public marketing/docs remain readable.

**Done when:** Overview and the prompt catalog demonstrate the final hierarchy at desktop/laptop/mobile sizes, and all routes, workspace switching, theme switching, command navigation, and keyboard focus still work.

### 3. Redesign prompt and playground workbench

Move metadata out of the saved editor's main flow; prioritize template space; keep draft/version state and actions visible. Reorganize playground settings and outputs, including a wider multi-model mode. Retain existing tabs for versions, releases, integration, and settings.

**Done when:** edit → preview/run → save → evaluate is clear, and comparing models does not require scrolling between settings and their corresponding outputs.

### 4. Redesign dataset editing

Apply navigator/table/detail placement; consolidate revision/actions; retain advanced JSON and import preview. Use the common detail panel for case editing.

**Done when:** selecting/editing a case, importing cases, saving a revision, and evaluating that exact revision all work without losing draft fields or context.

### 5. Redesign evaluation setup, results, and comparison

Split the large workbench into focused history, setup, and detail components, sharing selection/URL state. Implement grouped setup with a persistent summary/action area, then case-table/detail and reference/candidate comparison layouts.

**Done when:** the seeded regression is visible immediately in comparison; incomplete runs and errors remain accurate; refresh/back preserve relevant run, comparison, and case selection.

### 6. Align remaining screens and finalize

Apply shared patterns to Settings and Benchmarks, verify both themes and responsive behavior, update `DESIGN_SYSTEM.md`, and capture real product screenshots.

**Done when:** frontend lint, type checks, affected interaction tests, and production build pass, plus authenticated browser validation of the complete demo workflow.

## Validation and scope boundaries

- Inspect around 1440px desktop, 1280px laptop, 768px tablet, and 390px mobile widths. Check both themes, overflow, long names/JSON, focus visibility, keyboard navigation, drawer focus restoration, sticky-area overlap, and reduced motion.
- Verify empty/loading/error/populated states, saving failures, archived objects, imports, pagination, incomplete/cancelled runs, incompatible comparisons, and project switches.
- Extend existing interaction tests only for meaningful changed behavior. Run the existing relevant test suites, lint, TypeScript checks, and build. Backend tests become necessary if APIs change.
- Prefer current APIs/models. Do not silently filter only the loaded page while presenting a global filter. Any new server filtering requires matching totals and tenant-scoped validation.
- Keep auth, revision/version semantics, model execution, API key scopes, and release behavior intact. Do not add tracing, arbitrary dashboards, collaboration, global search, new providers, or independent multi-draft playgrounds as part of this work.
- Land changes in reviewable slices. The first visual milestone is the shell + Overview + prompt catalog; the next is the prompt workbench; the final core milestone is dataset-to-evaluation comparison.

## Langfuse references

- [Layout components](https://github.com/langfuse/langfuse/tree/main/web/src/components/layouts): consistent page wrappers, headers, tabs, actions, and scrolling.
- [Design-system guidelines](https://github.com/langfuse/langfuse/blob/main/web/src/components/design-system/README.md): reusable presentation primitives and explicit variants.
- [Data table](https://github.com/langfuse/langfuse/blob/main/web/src/components/table/data-table.tsx): consistent table controls and contextual inspection.
- [Playground](https://langfuse.com/docs/prompt-management/features/playground): adjacent settings/output and side-by-side inspection.
- [Experiments via UI](https://langfuse.com/docs/evaluation/experiments/experiments-via-ui): reproducible configuration and dataset compatibility.

These are principle references. LLMForge's layouts should follow its own prompt/version/dataset workflows and capabilities.

## Implemented core redesign — October 2026

- Compact workspace shell, page headers, controls, table toolbars, tabs, and pagination are shared across the application. Public pages retain their existing layout.
- Saved prompt metadata lives in Settings. The editor pairs with a playground that groups each model's settings and output; multiple models expand the playground area.
- Datasets use a navigator, action toolbar, revision disclosure, case table, and adjacent case editor. Narrow screens stack inspection below the table.
- Evaluation setup groups prompt/dataset, generation, and checks, with a persistent run summary and Start action. Results pair the case table with detail; comparisons align reference/candidate summaries and outputs. Selected case and filter survive refresh through URL state.
- Overview and Settings use compact shared panels. Benchmarks inherit the shell and shared presentation; their specialized statistical workflow remains intact.
- Drawer and command palette trap keyboard focus and restore it on close. Both themes, desktop/laptop/tablet/mobile placement, and page overflow were checked in the authenticated application.

Validation: frontend lint, all 41 tests, and the production build (including TypeScript) pass. In a separate Design review project, explicit mock generation verified an Echo demo v1 run with two passing cases and a v2 comparison with two regressions. The selected Farewell case and regression filter persisted after refresh. No live provider calls were needed.

Optional follow-up work: output text diffs, further decomposition of evaluation history/setup into separate components, and bespoke benchmark layout refinements. These are outside the completed core placement changes.
