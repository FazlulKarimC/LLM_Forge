# LLMForge — Frontend Design System

> **Stack:** Next.js 16 · Tailwind CSS v4 · Framer Motion · Lucide Icons  
> **Source of truth:** [`globals.css`](frontend/src/app/globals.css) · [`primitives.tsx`](frontend/src/components/ui/primitives.tsx)  
> **Theme:** Dark-first charcoal surfaces with warm gold actions and teal accents, shared by the workspace and public pages

The application shell prioritizes a compact workbench: Overview, Prompts, Datasets, and Evaluations form the main navigation; Benchmarks is separate; Settings and Docs are utilities. Public pages share the same identity, controls, flat panels and theme. `.app-shell` retains wider content, compact controls and smaller headings; `.public-shell` uses more spacious narrative layouts and larger touch targets.

---

## 🤖 Rules for LLM Agents & Developers

1. **Use CSS variables only.** Never hard-code hex/oklch values or generic Tailwind colours (`text-gray-500`, `bg-blue-600`). Reference `var(--foreground)`, `var(--accent)`, etc.
2. **Use Tailwind v4 variable syntax.** Write `text-(--muted-foreground)` and `border-(--border)`, not the older `text-[var(--muted-foreground)]` bracket syntax.
3. **Use utility classes defined in `globals.css`.** Page layout, panels, buttons, inputs, chips, pills, alerts, tables, and empty states all have dedicated CSS classes. Do not reinvent them.
4. **Use React primitives from `primitives.tsx`.** Import `PageHeader`, `Panel`, `PanelHeader`, `StatusPill`, `MetricCard`, `AnimatedNumber`, `MetricBar`, `EmptyState`, `SkeletonBlock`, `Keycap` — never rebuild these.
5. **Merge classes via `cn()`.** Import `cn` from `@/lib/utils` (clsx + tailwind-merge) for conditional class composition.
6. **Prefer flex/grid `gap` over margin.** Spacing is managed with `gap-*`, never ad hoc `mb-*` between siblings.
7. **Match layout to the task.** Use full-width sections for narrative content and two-column or table/detail layouts for editing and case inspection. Keep controls visible near the data they affect.
8. **Keep one identity.** Use `Brand` / `BrandGlyph` from the shared brand component. The original sidebar flask, 32px bordered tile and wordmark are authoritative; do not substitute Activity icons, image logos or a different typeface.
9. **Keep one theme.** The root `ThemeProvider` owns dark/light state and preserves `llmforge.theme.light`. Use `ThemeToggle`; never add page-specific theme state. Storage denial must not break rendering or toggling.

## Public pages and product storytelling

- Landing, docs and authentication use `.public-shell`, the shared public header/footer, and the same tokens as the workspace. Header/footer and the collapsed sidebar use the same flask mark; the favicon renders the same glyph. Static favicon raster colors are the exception to runtime CSS-variable colors.
- The landing page gives equal emphasis to benchmarks, prompts and evaluations. Its sequence is promise → real product walkthrough → workflow → capabilities → verifiable evidence → FAQ → workspace entry.
- Public content uses the existing 1180px maximum width, 24px desktop / 16px mobile gutters, 80px desktop / 48px mobile section spacing, and larger readable headings. Uniform styling does not require the workspace's compact text size on marketing pages.
- Use screenshots of the real application with deliberately prepared public examples. Capture only product regions, excluding account information, provider credentials, private content and internal request identifiers. Keep dark/light versions and readable mobile crops together under `public/walkthrough`.
- Seeded benchmark measurements and Echo demo checks must be labelled as illustrations, never model-performance evidence. Explain what the screen helps a visitor decide; do not invent customers, adoption numbers or a winning model.
- The public walkthrough is static, read-only and makes no application API requests. Its tabs support arrows, Home/End, selected state and focus. Workspace entry retains sign-in; Create demo examples remains an explicit action on Overview.
- Prefer static backgrounds and subtle hover feedback. Honor reduced motion; avoid canvas shaders, continuous motion and word-by-word hero reveals. Preview assets need useful alt text, dimensions and captions, and must remain readable in both themes.

---

## 🎨 Design Tokens

### Colour Palette (OKLCH)

| Token | Dark | Light | Usage |
|-------|------|-------|-------|
| `--background` | `oklch(0.145 0.008 255)` | `oklch(0.985 0.002 255)` | Page canvas |
| `--foreground` | `oklch(0.94 0.004 255)` | `oklch(0.23 0.01 255)` | Primary text / headings |
| `--surface-1` | `oklch(0.18 0.01 255)` | `oklch(0.965 0.004 255)` | Panel / card fill |
| `--surface-2` | `oklch(0.22 0.012 255)` | `oklch(0.94 0.006 255)` | Nested card / input fill |
| `--surface-3` | `oklch(0.26 0.012 255)` | `oklch(0.91 0.008 255)` | Hover fill |
| `--primary` | `oklch(0.79 0.09 84)` | *(same)* | Primary buttons, CTA accent (warm gold) |
| `--accent` | `oklch(0.74 0.08 182)` | `oklch(0.45 0.08 182)` | Links, focus rings, running-state accent (teal) |
| `--muted-foreground` | `oklch(0.71 0.01 255)` | `oklch(0.49 0.01 255)` | Descriptions, secondary text |
| `--border` | `oklch(0.31 0.01 255 / 0.9)` | `oklch(0.86 0.007 255 / 0.95)` | Default border |
| `--border-strong` | `oklch(0.44 0.012 255 / 0.95)` | `oklch(0.74 0.01 255 / 0.98)` | Hover / active border |
| `--destructive` | `oklch(0.67 0.18 28)` | `oklch(0.5 0.18 28)` | Error / danger |
| `--success` | `oklch(0.75 0.11 156)` | `oklch(0.45 0.11 156)` | Completed / pass |
| `--warning` | `oklch(0.82 0.09 84)` | `oklch(0.48 0.09 84)` | Queued / caution |

**Theme switching:** Controlled via `data-theme="light"` on `<html>`. Surface, border, muted and shadow values change along with darker semantic text shades in light mode.

**`color-mix` pattern:** For soft tints, use `color-mix(in oklab, var(--accent) 16%, transparent)`. This is the standard approach for status backgrounds, hover states, and soft highlights.

### Typography

| Token | Stack | Usage |
|-------|-------|-------|
| `--font-display` | Aptos Display → Segoe UI Variable Display → sans-serif | Page titles (`page-title`), `h1`–`h6` |
| `--font-body` | Aptos → Segoe UI Variable Text → sans-serif | All body text, buttons, labels |
| `--font-mono` | Cascadia Mono → Consolas → monospace | Metric values, code, timestamps |

Headings automatically apply `font-family: var(--font-display)` and `letter-spacing: -0.03em`.

### Sizing & Radius

| Element | Radius | Note |
|---------|--------|------|
| Page header | Flat | Context and title row |
| Panel / card | `10px` | Shared flat content wrapper |
| Empty state | `10px` | Shared panel shape |
| Inner card / metric-card | `10px` | Shared panel shape; prefer separators for nested groups |
| Code panel | `16px` | Monospace content |
| Button | `7px` | Shared control shape; public touch targets ≥44px, workspace ≥36px (40px on mobile) |
| Input | `7px` in workspace | Interactive workspace fields |
| Filmstrip cell | `12px` | Small clickable cells |
| Chip / pill / badge | `999px` | Fully rounded |

### Shadows

| Token | Usage |
|-------|-------|
| `--shadow-panel` | Panels, page headers, cards |
| `--shadow-overlay` | Modals, command palette |

### Layout Constants

| Token | Value | Usage |
|-------|-------|-------|
| `--sidebar-width` | `232px` | Sidebar nav width |
| `.page-width` | `min(100%, 1180px)` generally; up to `1600px` in `.app-shell` | Content max-width, centred |
| `.page-stack` | `flex-direction: column; gap: 1.5rem` | Vertical page content flow |

---

## 🧱 CSS Utility Classes

### Page Structure

| Class | Purpose |
|-------|---------|
| `.app-shell` | Root layout container, `min-height: 100vh` |
| `.page-width` | Centred content wrapper; wider inside `.app-shell` |
| `.page-stack` | Column flex with `1.5rem` gap for page sections |
| `.page-header` | Top-of-page block; compact and low-shadow inside `.app-shell` |
| `.page-header-row` | Flex row for title + inline actions (justify between) |
| `.page-eyebrow` | Small uppercase pill above title |
| `.page-title` | Responsive heading: `clamp(2rem, 4vw, 3.5rem)` |
| `.page-description` | Muted body text, `max-width: 46rem` |

### Panels

| Class | Purpose |
|-------|---------|
| `.panel` | Primary content card: border, `10px` radius, flat surface background |
| `.panel-muted` | Panel variant with darker surface background |
| `.panel-header` | Flex row header inside panel (title + actions) |
| `.panel-body` | Inner padding zone: `1.25rem` all sides |

### Buttons

All buttons share a `7px` radius, token-based colors, no glow or lifting motion, and a `160ms` color/border transition. Public buttons use `min-height: 2.8rem`; workspace density uses smaller controls.

| Class | Appearance | Hover |
|-------|-----------|-------|
| `.btn-primary` | Gold fill, dark text | Subtle brighten |
| `.btn-secondary` | Surface fill, border, foreground text | Stronger border |
| `.btn-ghost` | Transparent, muted text | Surface fill on hover |
| `.btn-danger` | Destructive tint, red text | Darker tint |

All support `:disabled` state with `opacity: 0.55` and no transform.

**Icon pattern:** Place a Lucide icon before the label text. The button's `gap: 0.6rem` handles spacing automatically.

```tsx
<button className="btn-primary">
  <Play className="size-4" />
  Run experiment
</button>
```

### Form Inputs

| Class | Element | Note |
|-------|---------|------|
| `.input-shell` | `<input>` | Full width; workspace 36px minimum desktop/tablet, 40px below 640px; public ≥44px; 7px radius |
| `.select-shell` | `<select>` | Same styling as input |
| `.textarea-shell` | `<textarea>` | `7rem` min-height, resizable |
| `.field-label` | `<label>` | Bold `0.88rem`, flex with space-between |
| `.field-help` | `<span>` | Muted `0.8rem` helper text |

**Focus ring:** Inputs use an accent border and a visible 2px ring. Reuse the shared input classes.

### Chips & Status Pills

| Class | Purpose |
|-------|---------|
| `.chip` | Metadata tag (model name, dataset, method) — pill-shaped, border, muted |
| `.status-pill` | Status indicator with icon — colour-coded by state |
| `.status-pending` | Grey |
| `.status-queued` | Amber/warning |
| `.status-running` | Teal/accent |
| `.status-completed` | Green/success |
| `.status-failed` | Red/destructive |

### Metrics

| Class | Purpose |
|-------|---------|
| `.metric-value` | Monospace, tabular-nums, tight tracking |
| `.metric-card` | Bordered card containing label + value + optional caption |
| `.metric-label` | Uppercase muted label (`0.78rem`) |
| `.metric-caption` | Muted body text below the value |

### Data Tables

| Class | Purpose |
|-------|---------|
| `.data-table` | Full-width table with collapsed borders |
| `thead th` | Uppercase bold labels with bottom border |
| `tbody td` | Padded cells with subtle border |
| `.data-row` | Table row with hover highlight transition |

### Feedback & States

| Class | Purpose |
|-------|---------|
| `.empty-state` | Dashed border box with icon, title, description, optional action |
| `.empty-state-icon` | `3rem` square icon container with accent tint |
| `.alert` | Horizontal flex: icon + message |
| `.alert-danger` | Red-tinted alert (errors) |
| `.alert-warning` | Amber-tinted alert (warnings) |
| `.alert-info` | Teal-tinted alert (info) |
| `.code-panel` | Monospace scrollable code block with border |
| `.skeleton` | Loading placeholder with shimmer animation |
| `.mono-caption` | Monospace timestamp/caption text |

---

## ⚛️ React Component API

All imported from `@/components/ui/primitives`.

### `PageHeader`

Top-of-page hero block. Use for every route.

```tsx
<PageHeader
  eyebrow={<><Icon className="size-3.5" /> Label</>}
  title="Page Title"
  description="One-line description."
  actions={<button className="btn-primary">Action</button>}  // optional, renders beside title
>
  {/* children render below title — status pills, action buttons */}
</PageHeader>
```

**Layout note:** For pages where the title can be very long (e.g. experiment detail), put action buttons in `children` instead of `actions` to prevent overflow.

### `Panel` + `PanelHeader`

Primary content wrapper. Every content section uses this.

```tsx
<Panel>
  <PanelHeader
    label="UPPERCASE LABEL"
    title="Section title"
    description="Muted one-liner."
    actions={<button className="btn-secondary">Action</button>}
  />
  <div className="panel-body">
    {/* content */}
  </div>
</Panel>
```

### `StatusPill`

Renders a colour-coded status badge with icon.

```tsx
<StatusPill status="completed" />  // "pending" | "queued" | "running" | "completed" | "failed"
```

### `MetricCard`

KPI display card with animated value.

```tsx
<MetricCard
  label="Exact accuracy"
  tone="success"        // "default" | "accent" | "success" | "warning" | "danger"
  value={<AnimatedNumber value={85.3} suffix="%" className="text-4xl" />}
  detail="42/50 runs marked correct"
/>
```

### `AnimatedNumber`

Spring-animated numeric display.

```tsx
<AnimatedNumber value={95.2} decimals={1} prefix="$" suffix=" ms" className="text-3xl" />
```

### `MetricBar`

Horizontal progress bar with animated fill.

```tsx
<MetricBar value={72.5} />  // 0-100
```

### `EmptyState`

Fallback for empty views or pending data.

```tsx
<EmptyState
  icon={<ScanSearch className="size-5" />}
  title="No results yet"
  description="Start the experiment to see metrics here."
  action={<button className="btn-primary">Run now</button>}
/>
```

### `SkeletonBlock`

Loading placeholder with shimmer.

```tsx
<SkeletonBlock className="h-[180px]" />
```

### `Keycap`

Keyboard shortcut indicator.

```tsx
<Keycap>Ctrl</Keycap> <Keycap>K</Keycap>
```

---

## 📐 Page Layout Conventions

### General Pattern

Every page follows this vertical stack structure:

```tsx
<div className="page-stack">
  <PageHeader ... />

  <section>
    <Panel> ... </Panel>
  </section>

  <section>
    <Panel> ... </Panel>
  </section>
</div>
```

Each `<section>` wraps one `<Panel>` (or a set of related panels). Use `space-y-4` inside `<div>` wrappers when nesting multiple panels in one logical group.

### Filters Above Content

Filters (dropdowns, search, status selectors) are always rendered as a **horizontal bar above the content** they filter. Never use a side column for filters.

```tsx
{/* Filters panel */}
<section>
  <Panel>
    <PanelHeader label="Filters" title="..." actions={<button>Clear</button>} />
    <div className="panel-body">
      <div className="flex flex-wrap items-end gap-4">
        <div className="min-w-[160px] flex-1 space-y-2">
          <label className="field-label">Status</label>
          <select className="select-shell">...</select>
        </div>
        {/* more filter fields */}
      </div>
    </div>
  </Panel>
</section>

{/* Full-width content */}
<section>
  <Panel>
    <PanelHeader label="Catalog" title="..." />
    <div className="panel-body">...</div>
  </Panel>
</section>
```

### Horizontal Metadata Cards

Small metadata items (lifecycle timestamps, readiness checks, system status) go in a responsive horizontal grid:

```tsx
<div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
  <div className="metric-card">...</div>
  <div className="metric-card">...</div>
  ...
</div>
```

### Text Overflow

| Context | Rule |
|---------|------|
| **List/catalog** items | Truncate names with `truncate`, cap descriptions at 2 lines with `line-clamp-2` |
| **Detail page** titles | Never truncate — show the full experiment name and description |
| **Filmstrip cells** | `min-w-0` + `truncate` + `overflow-hidden` on the status-pill text |
| **Action buttons** | Use `shrink-0` to prevent button compression |

For truncation to work, the parent must constrain width: `min-w-0 flex-1` on the text container.

---

## 🧭 App Shell & Navigation

The app shell (`app-shell.tsx`) provides:

- **Collapsible sidebar** (`232px` wide) with persistent collapse state via `localStorage`
- **Independent scroll** — sidebar and main content scroll independently (`overflow-y-auto` on both, `position: sticky` on sidebar)
- **Active state** — sidebar links match on exact path or path prefix, with priority to longer (more specific) matches
- **Header bar** with search/command palette trigger and external links
- **Theme toggle** — switches `data-theme` between `"light"` and blank (dark)

---

## ♿ Accessibility

- All interactive elements have `focus-visible` styles (accent ring + glow)
- Experiment cards use `role="button"`, `tabIndex={0}`, and `onKeyDown` for Enter/Space
- Status pills include a Lucide icon alongside text for colour-independent meaning
- Loading states use `animate-spin` on `LoaderCircle` icons
- Screen-reader utility: `.sr-only` class available for visually-hidden labels
- Empty states always include descriptive text and an action CTA

---

## 🔮 Micro-Animations

| Element | Animation | Easing |
|---------|-----------|--------|
| Buttons | Color and border only; no movement | 160ms |
| Metric values | requestAnimationFrame ease-out, 700ms; immediate final value with reduced motion | Cubic ease-out |
| Metric bars | Width grow from `0%` to target | Same spring curve, 0.7s |
| Skeleton shimmer | Gradient slide left→right | Linear, 1.4s loop |
| Data rows | Background-colour on hover | 160ms spring |
| Page background | Flat token-based canvas | Static |
| Overlays | Short fade/movement; no entrance/exit movement with reduced motion | 160–180ms |

## Workspace placement (October 2026)

The workspace uses compact density rules scoped to `.app-shell`. Public pages share its flat panels and control shapes, with larger narrative typography and touch targets. Workspace-specific layout and sizing overrides are listed below.

| Pattern | Workspace rule |
| --- | --- |
| Page identity | 25.6px desktop title, short context label, inline primary actions; no eyebrow card |
| Navigation | 232px sidebar, compact 36px rows, utilities anchored below the main workflow |
| Controls | 36px desktop minimum height, 40px on narrow screens, 7px radius; visible accent focus |
| Panels | 10px radius, flat surface, 16px internal padding |
| Collection | `TableToolbar` above a table, `PageControls` below; no pagination controls for a single page |
| Editor | `.prompt-workspace` pairs template and playground; comparison mode expands the playground |
| Tabs | `.workspace-tabs` and `.workspace-tab` use quiet underline selection with an accessible current-page state |
| Inputs | `.workbench-input` is shared by prompt/dataset/evaluation forms and Settings |
| Evaluation setup | Three grouped form sections; `.evaluation-summary` keeps configuration and Start visible |
| Case inspection | `.case-inspection` pairs table and `.case-detail` on wide screens and stacks them below 1100px |
| Feedback | Semantic status text and alerts; color alone never identifies an outcome |

Light mode uses darker semantic text shades for links, success, warning, errors, and focus accents. The warm primary button background remains distinct from the darker `--primary-text` used for inline links.

Saved prompt metadata belongs in Settings; template format, draft/saved state, and version notes belong with editing. Each playground model groups its provider/model controls with its last-run output. Shared generation parameters live in an expandable section. Runs compare an explicitly labeled candidate and reference; case filters and the inspected case are represented in the URL.

Do not add an application-wide filter that only filters one loaded page. Retain the distinction between drafts, immutable versions/revisions, and release labels. Use real demo data to verify result placement and keep provider credentials out of saved run configuration.

### Evaluation components

- Keep Runs and Evaluators within the Evaluations workspace, with clear links between them. Reuse `PageHeader`, `.table-toolbar`, `.data-table`, `.workbench-input`, shared panels and pagination controls; avoid a separate visual language for evaluator configuration.
- Put catalog search and archive filters above the evaluator table. Detail uses definition/version controls on the left and sample preview on the right, stacked on narrow screens. Score names, types, ranges/categories and passing rules need visible labels. Saving creates an immutable version; show stale-edit errors beside the preserved draft.
- New-run setup distinguishes inline checks from saved evaluator versions. Selected rows show name/version, required versus informational policy and explicit judge-key fields. Show effective generation settings and generation/judge call counts next to Start. Apply supported saved prompt settings only through an explicit action.
- Place saved-output scoring as a secondary expandable action on completed runs. Explain the new source-linked run and zero generation calls before starting. Existing outputs and prior scores remain accessible.
- Put candidate/reference controls and compatibility warnings above results. Reference search is server-filtered to completed runs on the same dataset revision and paginated independently of history. Preserve selected run/case/filter state in the URL.
- Score summaries show measured coverage, execution errors and missing values separately. Means exclude unscored cases; categorical values show distributions. Pair required/informational labels and pass/fail/error text with color. Numeric deltas require compatible scoring definitions; changed judge rubrics or evaluator versions must be visible.
- Keep generation and judge usage distinguishable in case detail. Provider-key inputs use password fields and request-only help text; previews do not enter run history. Loading, empty, validation and provider-error states must remain visible and accessible.

### Prompt management components

- Text/chat type is chosen during creation and remains visible, fixed, on saved prompts. Chat messages pair the role selector with the corresponding content field; reorder/remove controls stay on that message's row.
- Version configuration belongs in a disclosure inside the editor. Prompt tags belong in Settings; catalog tag/folder filters operate on the server before pagination.
- Versions places the saved history above an explicit Reference/Candidate comparison. Content and configuration changes have separate areas; color accompanies written added/removed labels. Comparison columns stack on narrow screens.
- Releases shows automatic latest separately from mutable staging, production and custom labels. Assigning a custom label uses the same saved-version confirmation as other promotions.
- Integration explains string versus chat compilation, version/config retrieval, and selector behavior. Additional API-key capabilities use explicit, unchecked-by-default controls in Settings.

## Durable evaluation presentation

### Session recovery and deployment discovery

Workspace bootstrap 401s and missing tokens show **Refresh session** and **Sign in
again** inside the shared `Panel`, with existing action tokens and visible busy
states. Refresh bypasses the cached session token. Sign in again ends the rejected
session and preserves the same-origin path/query/fragment. Transport and service
failures retain **Try again**. Private workspace content stays gated throughout.

The public guide distinguishes hosted HTTPS `/api/v1` consumption from local
backend setup. New evaluation setup and the guide show server-advertised generation
and judge provider choices, credential requirements, and the local Gemini/external
submission path. Discovery failures remain visible and do not claim provider
availability. Keep dashboard dataset/evaluator authoring steps separate from SDK
consumption and preserve external-score comparison warnings.

The guide's **Deployment identity** disclosure presents backend API/source/build
and frontend source/build separately. Missing identity is `unknown`; unavailable
endpoints are labeled unavailable. Show only designated release metadata, never
environment values, credentials, account data or project contents. Use the shared
theme and flask branding without adding a separate diagnostic visual style.

An interrupted execution retains public status **Failed** and displays an **Execution interrupted** recovery panel beside the selected run's summary. Use the warning border and existing text/input/button tokens. Explain preserved progress, pinned versions, remaining generation/judge calls, and the possibility of charging again for a lost provider response. Use the precise action **Resume unfinished work**. Do not offer resume for completed, cancelled or ordinary failed runs.

Credential inputs are password fields with autocomplete off and stay in local state. Request only the generation/inline-judge/pinned evaluator keys required by the run, and clear them on accepted resume. Retain the dispatch identity across an uncertain response. Show safe failure codes and attempt history in a compact disclosure; never show stored credentials or raw platform exceptions.

Run summary polling stops at terminal states. Case tables use bounded output previews, explicit totals, filters and previous/next pagination. Full input/output/check reasons load in the focused case panel when selected. Comparison totals come from the server; preserve policy compatibility warnings. Loading/error states must keep navigation and summary stable. Shared selections belong in URL parameters; project switching clears server caches and discards mounted credentials according to existing guards.


## Audit implementation conventions (October 2026)

The canonical workspace dimensions are a 232px sidebar, 36px desktop/tablet controls, 40px mobile controls, 7px control radius, 10px panel radius and 16px overlay radius. These are CSS tokens; public controls retain ≥44px touch targets. Light mode also changes semantic text shades (`--primary-text`, accent, success, warning and danger), not only surfaces.

Evaluation setup keeps the 280px desktop summary and a compact sticky version/count/Start summary below 1100px. Settings/limits and provider help expand on request. Prompt editor/playground remain mounted behind jump links on narrow screens; template minima are 340px desktop, 280px tablet and 220px mobile with manual vertical resizing.

Results keep compatibility/coverage full-width above a single table/detail pair at ≥1100px, stacked below that; a standalone detail fills the available width. Dataset selection has a border, tint and `aria-pressed`; the saved name precedes revision metadata and the mobile chooser collapses after selection.

Semantic tables use labelled rows below 640px, primary identity and status/progress/production first, with secondary metadata in `TableMetadata` disclosures. Scroll containers are labelled and keyboard focusable. Numeric desktop columns are right-aligned with tabular digits. `PageControls` hides single-page navigation while counts remain visible.

Loading, errors, first-use empty and filtered empty states are distinct. Filtered empty states offer Clear filters. Visible copy says Benchmark throughout; `/experiments` routes and API identifiers remain stable. Benchmark setup keeps required name/model/dataset/reasoning visible, tuning/routing/regression/lineage in disclosures, and a sticky summary distinguishing Save draft from Create and run.

JavaScript counters and overlays honor reduced motion as well as CSS. Icon-only actions have accessible names; decorative SVGs are hidden from assistive technology. Already-assigned releases are passive while production changes retain confirmation. Validate changes at 1440×900, 768×1024 and 390×844 in both themes, including long labels, focus and empty/selected states. Clearly distinguish local fixture verification from authenticated hosted verification.
