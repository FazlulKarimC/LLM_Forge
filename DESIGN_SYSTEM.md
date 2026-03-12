# LLMForge — Frontend Design System

> **Stack:** Next.js 16 · Tailwind CSS v4 · Framer Motion · Lucide Icons  
> **Source of truth:** [`globals.css`](frontend/src/app/globals.css) · [`primitives.tsx`](frontend/src/components/ui/primitives.tsx)  
> **Theme:** Dark-first editorial tech — OKLCH colour, glass surfaces, subtle grid texture

---

## 🤖 Rules for LLM Agents & Developers

1. **Use CSS variables only.** Never hard-code hex/oklch values or generic Tailwind colours (`text-gray-500`, `bg-blue-600`). Reference `var(--foreground)`, `var(--accent)`, etc.
2. **Use Tailwind v4 variable syntax.** Write `text-(--muted-foreground)` and `border-(--border)`, not the older `text-[var(--muted-foreground)]` bracket syntax.
3. **Use utility classes defined in `globals.css`.** Page layout, panels, buttons, inputs, chips, pills, alerts, tables, and empty states all have dedicated CSS classes. Do not reinvent them.
4. **Use React primitives from `primitives.tsx`.** Import `PageHeader`, `Panel`, `PanelHeader`, `StatusPill`, `MetricCard`, `AnimatedNumber`, `MetricBar`, `EmptyState`, `SkeletonBlock`, `Keycap` — never rebuild these.
5. **Merge classes via `cn()`.** Import `cn` from `@/lib/utils` (clsx + tailwind-merge) for conditional class composition.
6. **Prefer flex/grid `gap` over margin.** Spacing is managed with `gap-*`, never ad hoc `mb-*` between siblings.
7. **Favour stacked layouts.** Page sections should stack vertically in full-width `<section>` elements. Avoid side-by-side two-column grids unless the interaction pattern demands it (e.g. click-to-inspect filmstrip).

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
| `--accent` | `oklch(0.74 0.08 182)` | *(same)* | Links, focus rings, running-state accent (teal) |
| `--muted-foreground` | `oklch(0.71 0.01 255)` | `oklch(0.49 0.01 255)` | Descriptions, secondary text |
| `--border` | `oklch(0.31 0.01 255 / 0.9)` | `oklch(0.86 0.007 255 / 0.95)` | Default border |
| `--border-strong` | `oklch(0.44 0.012 255 / 0.95)` | `oklch(0.74 0.01 255 / 0.98)` | Hover / active border |
| `--destructive` | `oklch(0.67 0.18 28)` | *(same)* | Error / danger |
| `--success` | `oklch(0.75 0.11 156)` | *(same)* | Completed / pass |
| `--warning` | `oklch(0.82 0.09 84)` | *(same)* | Queued / caution |

**Theme switching:** Controlled via `data-theme="light"` on `<html>`. Only surface, border, muted, and shadow values change — semantic colours persist.

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
| Page header | `24px` | Outermost container |
| Panel / card | `22px` | Primary content wrapper |
| Empty state | `20px` | Dashed border variant |
| Inner card / metric-card | `18px` | Nested inside panels |
| Code panel | `16px` | Monospace content |
| Button / input | `14px` | Interactive elements |
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
| `--sidebar-width` | `264px` | Sidebar nav width |
| `.page-width` | `min(100%, 1180px)` | Content max-width, centred |
| `.page-stack` | `flex-direction: column; gap: 1.5rem` | Vertical page content flow |

---

## 🧱 CSS Utility Classes

### Page Structure

| Class | Purpose |
|-------|---------|
| `.app-shell` | Root layout container, `min-height: 100vh` |
| `.page-width` | Centred content wrapper, `max-width: 1180px` |
| `.page-stack` | Column flex with `1.5rem` gap for page sections |
| `.page-header` | Top-of-page hero block with border, shadow, radius `24px` |
| `.page-header-row` | Flex row for title + inline actions (justify between) |
| `.page-eyebrow` | Small uppercase pill above title |
| `.page-title` | Responsive heading: `clamp(2rem, 4vw, 3.5rem)` |
| `.page-description` | Muted body text, `max-width: 46rem` |

### Panels

| Class | Purpose |
|-------|---------|
| `.panel` | Primary content card: border, `22px` radius, glass background |
| `.panel-muted` | Panel variant with darker surface background |
| `.panel-header` | Flex row header inside panel (title + actions) |
| `.panel-body` | Inner padding zone: `1.25rem` all sides |

### Buttons

All buttons share: `min-height: 2.8rem`, `14px` radius, `160ms` spring transition, `gap: 0.6rem` for icon + text.

| Class | Appearance | Hover |
|-------|-----------|-------|
| `.btn-primary` | Gold fill, dark text, glow shadow | Lift + brighten |
| `.btn-secondary` | Surface fill, border, white text | Lift + stronger border |
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
| `.input-shell` | `<input>` | Full width, `14px` radius, `2.9rem` min-height |
| `.select-shell` | `<select>` | Same styling as input |
| `.textarea-shell` | `<textarea>` | `7rem` min-height, resizable |
| `.field-label` | `<label>` | Bold `0.88rem`, flex with space-between |
| `.field-help` | `<span>` | Muted `0.8rem` helper text |

**Focus ring:** All inputs show a double-ring focus: `1px accent border` + `6px accent glow`.

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

- **Collapsible sidebar** (`264px` wide) with persistent collapse state via `localStorage`
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
| Buttons | `translateY(-1px)` on hover | `cubic-bezier(0.16, 1, 0.3, 1)` |
| Metric values | Spring counter via `framer-motion` `animate()` | `[0.16, 1, 0.3, 1]` |
| Metric bars | Width grow from `0%` to target | Same spring curve, 0.7s |
| Skeleton shimmer | Gradient slide left→right | Linear, 1.4s loop |
| Data rows | Background-colour on hover | 160ms spring |
| Page background | Radial gradients (top-left gold, top-right teal) | Static |
| Grid overlay | 24px dot grid, fades radially | Static, `opacity: 0.1` |
