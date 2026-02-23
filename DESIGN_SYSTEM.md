# LlmForge - Frontend Design System

> **Based on:** Brillance SaaS Design | **Component Library:** shadcn/ui | **Styling:** Tailwind CSS v4

---

## 🤖 Instructions for LLM Agents

When generating frontend code for LlmForge, you **MUST** adhere to the following rules:

1. **Strict Palette Adherence:** ONLY use the 4 colors defined in the [Color Palette](#color-palette) (or semantic variants defined in CSS vars). Do not invent new hex codes or use generic Tailwind colors (e.g., `text-gray-500`, `bg-blue-600`) unless explicitly required for status indicators (success/warning/error).
2. **Typography Rules:** Use `font-serif` (Instrument Serif) ONLY for high-level headings/titles. Use `font-sans` (Inter) for all body text, UI elements, and labels.
3. **Component Reusability:** Favor `shadcn/ui` components located in `src/components/ui`. Look at existing implementations (like `Card`, `Button`, `Badge`) before writing raw HTML/CSS.
4. **Spacing & Layout:** Use `gap-*` for flex/grid spacing instead of margins. Default container width is `max-w-[1060px]`.
5. **Class Utilities:** Use `clsx` and `tailwind-merge` (`cn` utility) when combining dynamic class names.

---

## 🎨 Token Reference

### Color Palette (The 4-Color System)

| Token | HEX | CSS Variable | Usage |
|-------|-----|--------------|-------|
| **Dark Brown** | `#37322F` | `--primary` | Headings, primary buttons, primary text |
| **Off-White** | `#F7F5F3` | `--background` | Page background, secondary accents |
| **Taupe Gray** | `#605A57` | `--muted-foreground` | Body text, descriptions, secondary text |
| **Light Gray** | `#E0DEDB` | `--border` | Borders, dividers, subtle backgrounds |

*Note: Semantic status colors (`--success`, `--warning`, `--error`) are the only exceptions to this 4-color rule.*

### Typography

| Family | Tailwind Class | Usage |
|--------|----------------|-------|
| **Instrument Serif** | `font-serif` | Headlines, Page Titles, Card Titles |
| **Inter** | `font-sans` | Body text, UI elements, Labels, small text |

### Layout & Sizing

| Property | Value | Tailwind Class |
|----------|-------|----------------|
| **Container Width** | `1060px` | `max-w-[1060px]` |
| **Standard Gap** | `24px` | `gap-6` |
| **Default Radius**| `8px` | `rounded-lg` (Cards) |
| **Pill Radius** | `999px` | `rounded-full` (Buttons, Badges) |
| **Subtle Shadow** | - | `shadow-sm` (Elevated Cards) |

---

## 🧱 Component Patterns

When building or updating UI, follow these established structural patterns:

### Layout Container
```tsx
<div className="max-w-[1060px] mx-auto px-4 md:px-6 lg:px-0">
  {/* Content goes here */}
</div>
```

### Typography Hierarchy
```tsx
<h1 className="text-[36px] md:text-[52px] font-serif leading-tight text-[#37322F]">Page Title</h1>
<h2 className="text-2xl font-serif text-[#37322F]">Section Heading</h2>
<p className="text-base text-[#605A57]">Standard body copy uses Taupe Gray and Inter.</p>
```

### Action Interactions
- **Primary Action (Dark):** `bg-[#37322F] text-white hover:bg-[#2A2520] rounded-full`
- **Secondary Action (Light):** `bg-white border-[#E0DEDB] text-[#37322F] rounded-full`

## ♿ Accessibility

- All interactive elements must maintain sufficient focus states (e.g., `focus-visible:ring-2 focus-visible:ring-[#37322F]/20`).
- Ensure `aria-*` tags and `sr-only` generic semantic descriptors are used for screen readers on dynamic components (Loading Spinners, Status Alerts).
