# Frontend design guide

This is the single maintained guide for LLMForge frontend design and interaction. Update it alongside changes to shared components or layout behavior. Setup and API guides describe product contracts; they should link here for design rules. Completed plans, audit reports and screenshots are historical evidence in Git, not competing design specifications.

## Implementation sources

Use the existing Next.js, React, Tailwind CSS v4, Framer Motion and Lucide stack. The implementation is authoritative when exact values or component props are needed:

- [Global styles and tokens](frontend/src/app/globals.css)
- [Shared primitives](frontend/src/components/ui/primitives.tsx), [pagination](frontend/src/components/ui/page-controls.tsx), and [table metadata](frontend/src/components/ui/table-metadata.tsx)
- [Application shell](frontend/src/components/app-shell.tsx), [brand](frontend/src/components/brand.tsx), and [theme provider](frontend/src/components/theme-provider.tsx)
- Domain workbenches in [features](frontend/src/features) and benchmark routes under [experiments](frontend/src/app/(app)/experiments)

## Identity and styling

Keep the shared flask glyph, 32px bordered brand tile and wordmark. Use `Brand` and `BrandGlyph`; keep the favicon aligned with them. Use flat charcoal surfaces, warm gold primary actions and teal interaction accents. Public pages and authenticated workspaces share this identity and both themes.

Use CSS variables for colors. Prefer Tailwind v4 syntax such as `text-(--muted-foreground)` and `border-(--border)`; use `cn()` from `@/lib/utils` for conditional classes. Use `color-mix(in oklab, var(--accent) 16%, transparent)` for soft tints. Avoid introducing another palette, styling framework, logo, continuous background animation or hover movement.

| Token group | Purpose |
| --- | --- |
| `--background`, `--foreground` | Canvas and primary text |
| `--surface-1`, `--surface-2`, `--surface-3` | Panels, nested/input surfaces and hover fills |
| `--primary`, `--primary-foreground` | Gold action fill and its dark label |
| `--primary-text` | Gold-family inline text; darkens in light mode |
| `--accent`, `--ring` | Links and visible focus |
| `--muted-foreground` | Supporting copy and metadata |
| `--success`, `--warning`, `--destructive` | Semantic status, paired with written meaning |
| `--border`, `--border-strong` | Boundaries and selected/hover emphasis |
| `--shadow-overlay` | Dialogs and command palette; ordinary panels stay flat |

The root `ThemeProvider` owns the theme through `data-theme="dark"` / `"light"` and the `llmforge.theme.light` preference. Use `ThemeToggle`, preserve the before-paint bootstrap, and tolerate denied storage. Do not add page-local theme state. Light mode changes semantic text and focus colors as well as backgrounds. Static favicon raster colors are an asset exception to runtime variables.

Use `--font-display` for headings (Aptos Display / Segoe UI fallbacks), `--font-body` for controls and prose (Aptos / Segoe UI), and `--font-mono` for code and metrics (Cascadia / Consolas). Use tabular digits for comparable numeric data. Avoid low-contrast tiny text for essential decisions.

## Dimensions and responsive behavior

These are effective workspace values, including overrides near the end of `globals.css`; do not copy earlier overridden declarations as new defaults.

| Pattern | Current rule |
| --- | --- |
| Sidebar | 232px expanded, 64px collapsed; desktop sidebar from 1024px, navigation drawer below |
| Workspace content | `.page-width` up to 1600px; 16px gutters, 20px from the `sm` breakpoint |
| Public content | Up to 1180px; 24px desktop / 16px mobile gutters |
| Workspace text | 14px base; page titles 1.6rem, 1.4rem below 640px |
| Workspace sections | `.page-stack` gap 1.25rem; panel content padding 1rem |
| Standard controls | `--workspace-control-height`: 36px desktop/tablet, 40px below 640px |
| Public controls | At least 44px touch targets; more generous narrative typography |
| Shapes | `--radius-control`: 7px; `--radius-panel`: 10px; `--radius-overlay`: 16px |
| Supporting shapes | Pills fully rounded; existing case table/detail boundaries 8px, brand tile 8px |
| Editor template | Vertically resizable; minimum 340px at 1100px+, 280px tablet, 220px below 640px |
| Split workbenches | Editor/playground and legacy case inspection split from 1100px; stack below |
| Multiple playground models | Columns from 1280px; wide playground mode gives outputs more space |

Preserve selection, drafts and focus as layouts change. Keep `min-w-0` on grid/flex children; wrap long detail titles and structured content. Truncate catalog previews only when the full value remains available in detail. Constrain code/output scrolling to its own region; prevent horizontal page overflow.

The global workspace bar is sticky. Evaluation summary offsets are 76px on desktop and 56px on narrower screens; benchmark actions use 64px desktop / 56px mobile. The prompt header is sticky only at 1100px+. Check combinations of sticky areas so actions, validation and focused content remain visible.

## Shared components and page patterns

Reuse the classes in `globals.css` and the shared React components before adding local variants. Prefer layout `gap` for sibling spacing and separators/disclosures for related groups; avoid unnecessary nested bordered panels.

| Building block | Use |
| --- | --- |
| `PageHeader` | Route identity, description, back link and actions; long-title actions may use children |
| `Panel`, `PanelHeader`, `.panel-body` | Related content and section actions; use semantic headings inside content where needed |
| `SectionHeading`, `TableToolbar` | Compact workbench headings and collection filters |
| `StatusPill` | Benchmark lifecycle badge; evaluations retain their domain status components |
| `MetricCard`, `AnimatedNumber`, `MetricBar` | Labelled metrics, measured values, captions and progress |
| `EmptyState`, `.collection-empty`, `SkeletonBlock` | Distinct first-use/filtered states and loading placeholders |
| `.btn-primary`, `.btn-secondary`, `.btn-ghost`, `.btn-danger` | Shared action hierarchy, disabled states and focus |
| `.input-shell`, `.select-shell`, `.textarea-shell`, `.workbench-input` | Labelled fields and consistent focus/density |
| `.workspace-tabs`, `.workspace-tab` | Quiet underline navigation; preserve current-page/selected semantics |
| `.alert-*`, `.code-panel` | Visible feedback and bounded code content |

A collection uses page identity/actions, then a wrapping search/filter toolbar, then table/content and pagination. A workbench places the editor or table beside contextual detail on wide screens. Settings uses bounded form groups. Put each action next to its subject and keep one clear primary action. Use existing native disclosures for advanced settings instead of always exposing uncommon controls.

Filters apply on the server before pagination when presented as global filters. Counts must describe that same filtered scope. Loading, errors, first-use empty and filtered empty states remain distinct; a filtered empty state offers **Clear filters**. Do not display an empty-success message for a failed request.

## Tables and case inspection

Use semantic `.data-table` tables. The prompt catalog, history and case views use `.responsive-table` to become labelled rows below 640px: each cell needs a meaningful `data-label`. Identity and status/progress/release information come first. Use `TableMetadata` for secondary information that appears inline on desktop and in a disclosure on mobile. Reuse `.numeric` for right-aligned, tabular desktop values; mobile labelled values align left.

Label overflow regions and make them keyboard focusable. Selected rows expose state as well as border/tint. Action controls retain the shared touch height. Keep result counts visible; `PageControls` hides navigation for a single page. Its default page size is 50; evaluation case pages explicitly use 25.

Results show compatibility/coverage across the full width above one `.case-inspection` table/detail pair. At 1100px+ it uses a flexible table and a detail column with a 340px minimum; below that it stacks. A standalone detail or single-run summary fills the available width. Full input/output/check reasons load when a case is selected; focus the loaded detail. Keep candidate/reference direction and case filters explicit, with selection/filter state in the URL.

## Workflow placement and safeguards

### Prompts

- Keep draft/saved state, selected immutable version and Save/Evaluate actions close to the editor. Saved name/description/tags belong in Settings; version notes and saved configuration use disclosures near editing.
- Keep editor and playground mounted when using narrow-screen jump links, so navigation cannot discard a draft. Each model groups its provider/model/key controls with its output. Shared generation parameters expand on request; independent drafts are outside the current playground contract.
- Choose text/chat type at creation and retain it across versions. Chat role/content and reorder/remove controls stay together. Versions shows explicit Reference/Candidate content and configuration changes; columns stack on narrow screens.
- Releases distinguishes automatic latest from staging, production and custom labels. Already-assigned targets are disabled **Assigned** states. Saving a version differs from promoting it; production and custom-label actions retain saved-version confirmation and dirty-draft guards.
- Integration explains compilation, pinned versions and label selection. Optional API-key capabilities remain explicit and unchecked by default in Settings.

### Evaluations

The visible workflow is SDK instructions → local execution → summary history and compatible reference summaries. There is no Datasets navigation, browser dataset editor/import or hosted new-run setup. Keep Benchmarks separate and visible.

New local reports show Local dataset, SDK execution, fingerprints, total/processed/pass/error counts and optional aggregate metrics. State that inputs/outputs/checks remain in the local artifact. Do not render empty case grids, host-side resume or saved-output scoring for summary-only runs. Fingerprints and scores are SDK-reported; matching identity is not independent verification.

Existing hosted results retain case inspection and appropriate recovery actions. Label those as legacy history. Reference controls must not pair two null dataset revision IDs; use the explicit source/fingerprint contract. Keep URL selection and stable loading/error states.

### Benchmarks

Visible copy says **Benchmark**, while existing `/experiments` routes and API identifiers remain stable. Keep statistical comparison separate from prompt evaluations. Setup shows required name/model/dataset/reasoning controls; tuning, routing, regression and lineage use disclosures. Keep the sticky model/dataset summary with **Save draft** and **Create and run**. Results retain provenance, partial-run state, exports and statistical caveats.

### Sessions, credentials and interrupted work

Keep private content gated on rejected sessions. Authentication recovery offers **Refresh session** and **Sign in again**, preserving the same-origin destination. Service/transport failures retain **Try again**. Project switching clears scoped caches and follows existing draft guards.

Provider credentials use password inputs, autocomplete off and request-only local state. Clear them after accepted operations and relevant provider/project changes; never put them in URLs, persisted configuration, exports or screenshots. Preserve the same dispatch identity after an uncertain response and show safe request IDs beside retry guidance.

An interrupted run retains public status **Failed** and an **Execution interrupted** warning beside its summary. Show preserved progress, pinned configuration, remaining calls and the possibility of repeating a charged response lost before checkpointing. Offer **Resume unfinished work** only for resumable interrupted attempts, requiring fresh keys; completed, cancelled and ordinary failures do not get that action. Attempt history belongs in a compact disclosure; raw platform exceptions do not.

Stop summary polling at terminal states. Retain navigation and summary while details load or fail. Deployment identity presents frontend and backend source/build separately; missing is unknown and unavailable endpoints stay labelled unavailable.

## Public pages

Landing, public docs and authentication use `.public-shell` with shared header/footer, theme and flask branding. Narrative sections use 80px desktop / 48px mobile vertical spacing. Explain the prompt/version/local-evaluation/release workflow and give Benchmarks its own clear entry.

Walkthroughs use real prepared examples, useful alt text, captions and dimensions. Keep dark/light assets and readable mobile crops under `frontend/public/walkthrough`; its README documents asset capture only and links here for design. Demo measurements are illustrations, not model-performance or adoption claims. Public walkthrough tabs are static/read-only with arrows, Home/End and focus semantics; workspace entry retains authentication. Separate hosted HTTPS API consumption from local backend setup.

## Accessibility and motion

Use labelled native buttons, links and inputs; icon-only controls need accessible names and decorative SVGs are hidden from assistive technology. Preserve focus rings, logical keyboard order, disclosure semantics, dialog focus trapping/Escape and focus restoration. Status text and icons must carry meaning independently of color. Keep errors near preserved inputs and use appropriate alert/status announcements.

Use short color/border transitions without lifting controls. `AnimatedNumber` uses a 700ms cubic ease-out RAF update, not a spring; `MetricBar` transitions width for 700ms. Both render final values immediately with reduced motion. Dialog/navigation overlays skip movement under the same preference. Honor reduced motion in CSS and JavaScript; avoid continuous decorative motion.

## Verification and maintenance

For frontend behavior changes, run relevant interaction tests, lint, TypeScript and production build as described in [testing](docs/DEVELOPMENT.md#testing). Documentation-only edits need link/source checks rather than a fresh application build.

Check 1440×900 desktop, 768×1024 tablet and 390×844 phone in dark and light themes; add 1280px when changing multi-model layout. Exercise long names/JSON, empty/loading/error/selected states, sticky overlap, keyboard focus, reduced motion, pagination/filter reset, drafts and project switches. Verify the provider-free prompt → saved version → fixed revision → comparison journey when changing those workflows. Distinguish fixture-based checks from authenticated hosted checks, and report unverified provider behavior accurately.

Keep this guide focused on current conventions. Put operational contracts in their domain guides and regenerate API schemas through the existing contract workflow. Store completed task history in commits instead of adding another frontend plan or design document.
