# LlmForge - Frontend Design System

> **Based on:** Brillance SaaS Design | **Component Library:** shadcn/ui | **Styling:** Tailwind CSS v4

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Color Palette](#color-palette)
3. [Typography](#typography)
4. [Spacing System](#spacing-system)
5. [Border Radius](#border-radius)
6. [Shadows](#shadows)
7. [Transitions](#transitions)
8. [Responsive Design](#responsive-design)
9. [shadcn/ui Components](#shadcnui-components)
10. [Loading & Empty States](#loading--empty-states)
11. [Data Visualization](#data-visualization)
12. [Code Display](#code-display)
13. [Common Patterns](#common-patterns)
14. [Icons](#icons)
15. [Accessibility](#accessibility)
16. [Do's and Don'ts](#dos-and-donts)
17. [Setup](#setup)

---

## Quick Start

```bash
# Install shadcn/ui
npx shadcn@latest init

# Add commonly used components
npx shadcn@latest add button card badge table tabs form input skeleton chart
```

---

## Color Palette

### 4-Color System

| Token | HEX | CSS Variable | Usage |
|-------|-----|--------------|-------|
| **Dark Brown** | `#37322F` | `--primary` | Headings, buttons, primary text |
| **Off-White** | `#F7F5F3` | `--background` | Page background |
| **Taupe Gray** | `#605A57` | `--muted-foreground` | Body text, descriptions |
| **Light Gray** | `#E0DEDB` | `--border` | Borders, dividers |

### Semantic Colors

```css
/* Text */
--text-heading: #37322F;
--text-body: #605A57;
--text-muted: rgba(55, 50, 47, 0.80);
--text-on-dark: #FBFAF9;

/* Backgrounds */
--bg-page: #F7F5F3;
--bg-card: #FFFFFF;
--bg-primary: #37322F;

/* Borders */
--border-standard: rgba(55, 50, 47, 0.12);
--border-subtle: rgba(55, 50, 47, 0.06);

/* Status Colors (only exceptions to 4-color rule) */
--success: #22c55e;
--warning: #eab308;
--error: #ef4444;
```

---

## Typography

### Font Stack

| Family | Usage | Variable |
|--------|-------|----------|
| **Instrument Serif** | Headlines, titles | `font-serif` |
| **Inter** | Body, UI, labels | `font-sans` |

### Type Scale

| Element | Classes |
|---------|---------|
| Hero Title | `text-[52px] md:text-[80px] font-serif leading-tight` |
| Section Title | `text-[36px] font-serif` |
| Card Title | `text-xl font-serif` |
| Body Large | `text-lg font-medium text-[#605A57]` |
| Body | `text-base text-[#605A57]` |
| Small/Label | `text-sm font-medium` |
| Badge/Code | `text-xs font-medium` |

---

## Spacing System

| Scale | Tailwind | Pixels | Usage |
|-------|----------|--------|-------|
| Micro | `gap-1, gap-2` | 4px, 8px | Icon-text gaps |
| Compact | `gap-3, gap-4` | 12px, 16px | Within cards |
| Standard | `gap-6` | 24px | Between elements |
| Generous | `gap-8, gap-12` | 32px, 48px | Page sections |
| Hero | `gap-16, gap-24` | 64px, 96px | Major separations |

### Container

```tsx
<div className="max-w-[1060px] mx-auto px-4 md:px-6 lg:px-0">
  {/* Page content */}
</div>
```

### Padding Rules

| Context | Mobile | Tablet | Desktop |
|---------|--------|--------|---------|
| Page | `px-4` | `px-6` | `px-0` (container) |
| Card | `p-4` | `p-6` | `p-6` |
| Section | `py-12` | `py-16` | `py-24` |

---

## Border Radius

| Element | Class | Value |
|---------|-------|-------|
| Buttons/Badges | `rounded-full` | 999px |
| Cards | `rounded-lg` | 8px |
| Inputs | `rounded-md` | 6px |
| Minimal | `rounded-[3px]` | 3px |

---

## Shadows

| Type | Class/Value | Usage |
|------|-------------|-------|
| Subtle | `shadow-xs` | Cards, dropdowns |
| Standard | `shadow-sm` | Elevated cards |
| Button Inset | `shadow-[0px_0px_0px_2.5px_rgba(255,255,255,0.08)_inset]` | Dark buttons |
| Elevation | `shadow-lg` | Modals, popovers |

---

## Transitions

```tsx
// Color changes (hover states)
className="transition-colors duration-200"

// All properties (smooth state changes)
className="transition-all duration-300 ease-in-out"

// Focus effects
className="transition-[color,box-shadow] duration-200"
```

---

## Responsive Design

### Breakpoints

| Breakpoint | Min-width | Prefix | Use Case |
|------------|-----------|--------|----------|
| Mobile | 0px | (default) | Phones |
| Small | 640px | `sm:` | Large phones |
| Medium | 768px | `md:` | Tablets |
| Large | 1024px | `lg:` | Desktops |
| XL | 1280px | `xl:` | Large monitors |

### Responsive Patterns

```tsx
// Stack → Row
className="flex flex-col md:flex-row gap-4 md:gap-8"

// Grid columns
className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"

// Responsive text
className="text-2xl md:text-3xl lg:text-4xl"

// Hide/show
className="hidden md:block"  // Hidden on mobile
className="md:hidden"         // Hidden on desktop
```

---

## shadcn/ui Components

### Buttons

```tsx
// Primary (Dark)
<Button className="bg-[#37322F] hover:bg-[#2A2520] text-white rounded-full px-8">
  Run Experiment
</Button>

// Secondary (White)
<Button variant="outline" className="bg-white border-[#E0DEDB] text-[#37322F] rounded-full">
  Cancel
</Button>

// Ghost
<Button variant="ghost" className="text-[#37322F] hover:bg-[#37322F]/5">
  View Details
</Button>
```

### Cards

```tsx
<Card className="bg-white border-[#E0DEDB] rounded-lg shadow-sm">
  <CardHeader>
    <CardTitle className="font-serif text-[#37322F]">Results</CardTitle>
  </CardHeader>
  <CardContent className="text-[#605A57]">
    {/* Content */}
  </CardContent>
</Card>
```

### Badges

```tsx
// Default
<Badge className="bg-white border border-[rgba(2,6,23,0.08)] text-[#37322F] rounded-full">
  CoT
</Badge>

// Status
<Badge className="bg-green-50 text-green-700 border-green-200">Completed</Badge>
<Badge className="bg-yellow-50 text-yellow-700 border-yellow-200">Running</Badge>
<Badge className="bg-red-50 text-red-700 border-red-200">Failed</Badge>
```

### Tables

```tsx
<Table>
  <TableHeader className="bg-[#F7F5F3] border-b border-[#E0DEDB]">
    <TableRow>
      <TableHead className="text-[#37322F] font-medium">Experiment</TableHead>
    </TableRow>
  </TableHeader>
  <TableBody>
    <TableRow className="border-b border-[rgba(55,50,47,0.12)] hover:bg-[#F7F5F3]/50">
      <TableCell className="text-[#605A57]">naive_vs_cot</TableCell>
    </TableRow>
  </TableBody>
</Table>
```

### Tabs

```tsx
<Tabs defaultValue="results">
  <TabsList className="bg-[#F7F5F3] border border-[#E0DEDB] rounded-lg p-1">
    <TabsTrigger 
      value="results" 
      className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-md"
    >
      Results
    </TabsTrigger>
  </TabsList>
</Tabs>
```

### Forms

```tsx
<Form {...form}>
  <form className="space-y-6">
    <FormField
      control={form.control}
      name="name"
      render={({ field }) => (
        <FormItem>
          <FormLabel className="text-sm font-medium text-[#37322F]">
            Experiment Name
          </FormLabel>
          <FormControl>
            <Input 
              className="h-9 border-[#E0DEDB] rounded-md text-base md:text-sm"
              placeholder="naive_vs_cot" 
              {...field} 
            />
          </FormControl>
          <FormDescription className="text-sm text-[#605A57]">
            Use snake_case for experiment names
          </FormDescription>
          <FormMessage className="text-sm text-red-500" />
        </FormItem>
      )}
    />
  </form>
</Form>
```

### Alerts/Toasts

```tsx
// Success
<Alert className="border-green-200 bg-green-50">
  <CheckCircle className="size-4 text-green-600" />
  <AlertTitle className="text-green-800">Experiment completed</AlertTitle>
  <AlertDescription className="text-green-700">
    Results are ready to view.
  </AlertDescription>
</Alert>

// Error
<Alert variant="destructive">
  <AlertCircle className="size-4" />
  <AlertTitle>Error</AlertTitle>
  <AlertDescription>Model failed to load.</AlertDescription>
</Alert>
```

### Progress

```tsx
// Determinate
<Progress value={75} className="h-2 bg-[#E0DEDB]" />

// Indeterminate (CSS animation)
<div className="h-2 bg-[#E0DEDB] rounded-full overflow-hidden">
  <div className="h-full bg-[#37322F] animate-progress" />
</div>
```

---

## Loading & Empty States

### Skeleton

```tsx
// Text skeleton
<div className="space-y-2">
  <Skeleton className="h-4 w-3/4" />
  <Skeleton className="h-4 w-1/2" />
</div>

// Card skeleton
<Skeleton className="h-64 w-full rounded-lg" />

// Avatar skeleton
<Skeleton className="size-10 rounded-full" />
```

### Loading Card

```tsx
function CardLoading() {
  return (
    <Card className="p-6">
      <Skeleton className="h-6 w-48 mb-4" />
      <div className="space-y-2">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6" />
        <Skeleton className="h-4 w-4/6" />
      </div>
    </Card>
  )
}
```

### Empty State

```tsx
function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="size-16 rounded-full bg-[#F7F5F3] flex items-center justify-center mb-4">
        <FlaskConical className="size-8 text-[#605A57]" />
      </div>
      <h3 className="font-serif text-xl text-[#37322F] mb-2">No experiments yet</h3>
      <p className="text-[#605A57] mb-6 max-w-sm">
        Create your first experiment to start comparing LLM methods.
      </p>
      <Button className="bg-[#37322F] text-white rounded-full">
        New Experiment
      </Button>
    </div>
  )
}
```

---

## Data Visualization

### Chart Colors

```tsx
const chartConfig = {
  accuracy: { label: "Accuracy", color: "#37322F" },
  latency: { label: "Latency", color: "#605A57" },
  naive: { label: "Naive", color: "#E0DEDB" },
  cot: { label: "CoT", color: "#37322F" },
}
```

### Line Chart

```tsx
<ChartContainer config={chartConfig} className="h-80">
  <LineChart data={data}>
    <CartesianGrid strokeDasharray="3 3" stroke="#E0DEDB" />
    <XAxis dataKey="name" stroke="#605A57" fontSize={12} />
    <YAxis stroke="#605A57" fontSize={12} />
    <ChartTooltip content={<ChartTooltipContent />} />
    <Line type="monotone" dataKey="accuracy" stroke="#37322F" strokeWidth={2} />
  </LineChart>
</ChartContainer>
```

### Bar Chart (Comparison)

```tsx
<ChartContainer config={chartConfig} className="h-64">
  <BarChart data={comparisonData}>
    <CartesianGrid strokeDasharray="3 3" stroke="#E0DEDB" />
    <XAxis dataKey="method" />
    <YAxis />
    <ChartTooltip content={<ChartTooltipContent />} />
    <Bar dataKey="accuracy" fill="#37322F" radius={[4, 4, 0, 0]} />
  </BarChart>
</ChartContainer>
```

---

## Code Display

### Inline Code

```tsx
<code className="px-1.5 py-0.5 bg-[#F7F5F3] text-[#37322F] text-sm font-mono rounded">
  naive_vs_cot
</code>
```

### Code Block

```tsx
<pre className="p-4 bg-[#37322F] text-[#F7F5F3] rounded-lg overflow-x-auto">
  <code className="text-sm font-mono">
    {`{
  "method": "chain_of_thought",
  "model": "microsoft/phi-2"
}`}
  </code>
</pre>
```

---

## Common Patterns

### Page Header

```tsx
<header className="border-b border-[rgba(55,50,47,0.12)] bg-[#F7F5F3]">
  <div className="max-w-[1060px] mx-auto px-4 py-4 flex justify-between items-center">
    <h1 className="font-serif text-xl text-[#37322F]">LlmForge</h1>
    <nav className="flex gap-6">
      <Link className="text-sm font-medium text-[#37322F] hover:text-[#37322F]/80">
        Experiments
      </Link>
    </nav>
  </div>
</header>
```

### Metric Card

```tsx
<Card className="bg-white border-[#E0DEDB]">
  <CardContent className="p-6">
    <p className="text-sm text-[#605A57]">Accuracy</p>
    <p className="text-3xl font-serif text-[#37322F]">58.2%</p>
    <p className="text-xs text-green-600">+16% from baseline</p>
  </CardContent>
</Card>
```

### Divider

```tsx
<div className="border-t border-[rgba(55,50,47,0.12)] my-8" />
```

### Section Header

```tsx
<div className="flex items-center justify-between mb-6">
  <h2 className="font-serif text-2xl text-[#37322F]">Experiments</h2>
  <Button className="bg-[#37322F] text-white rounded-full">
    New Experiment
  </Button>
</div>
```

---

## Icons

### Library: Lucide React

```bash
npm install lucide-react
```

### Sizing

| Size | Class | Pixels | Usage |
|------|-------|--------|-------|
| XS | `size-3` | 12px | Indicators |
| SM | `size-4` | 16px | Default |
| MD | `size-5` | 20px | Buttons |
| LG | `size-6` | 24px | Headers |

### Common Icons

```tsx
import { 
  FlaskConical,    // Experiments
  BarChart3,       // Results/Metrics
  Play,            // Run
  Settings,        // Config
  ChevronRight,    // Navigation
  Check,           // Success
  X,               // Close/Error
  Loader2,         // Loading (animate-spin)
} from 'lucide-react'
```

### Icon Button

```tsx
<Button variant="ghost" size="icon" className="size-9">
  <Settings className="size-4" />
  <span className="sr-only">Settings</span>
</Button>
```

---

## Accessibility

### Focus States

```tsx
className="focus:outline-none focus-visible:ring-2 focus-visible:ring-[#37322F]/20 focus-visible:ring-offset-2"
```

### Screen Reader

```tsx
// Hidden visually, readable by screen readers
<span className="sr-only">Loading experiments</span>

// Skip link
<a href="#main" className="sr-only focus:not-sr-only">Skip to content</a>
```

### ARIA

```tsx
// Loading state
<div aria-busy="true" aria-label="Loading experiments">

// Live regions
<div aria-live="polite" aria-atomic="true">
  {statusMessage}
</div>
```

### Color Contrast

- All text meets **WCAG AA** (4.5:1 minimum)
- Dark Brown `#37322F` on Off-White `#F7F5F3`: **8.2:1** ✓
- Taupe Gray `#605A57` on White: **5.1:1** ✓

---

## Do's and Don'ts

### ✅ DO

- Use only the 4-color palette (exception: status colors)
- Serif for headings, sans for body
- `gap` for spacing (not margins)
- Generous whitespace
- `rounded-full` for buttons/badges
- Mobile-first responsive design
- Test all interactive states

### ❌ DON'T

- Introduce new colors
- Mix font families incorrectly
- Use gradients
- Cram content together
- Use thick borders (prefer shadows)
- Forget responsive prefixes
- Ignore accessibility

---

## Setup

### globals.css

```css
@import "tailwindcss";

@theme {
  --font-sans: 'Inter', system-ui, sans-serif;
  --font-serif: 'Instrument Serif', Georgia, serif;
}

:root {
  --background: #F7F5F3;
  --foreground: #37322F;
  --muted-foreground: #605A57;
  --border: #E0DEDB;
  --card: #FFFFFF;
  --primary: #37322F;
  --primary-foreground: #FFFFFF;
}

body {
  @apply bg-[#F7F5F3] text-[#37322F] font-sans antialiased;
}
```

### layout.tsx (Fonts)

```tsx
import { Inter } from 'next/font/google'
import localFont from 'next/font/local'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })

const instrumentSerif = localFont({
  src: './fonts/InstrumentSerif-Regular.ttf',
  variable: '--font-serif',
})

export default function RootLayout({ children }) {
  return (
    <html className={`${inter.variable} ${instrumentSerif.variable}`}>
      <body>{children}</body>
    </html>
  )
}
```

---

## Quick Reference

| Component | Height | Rounded | Shadow |
|-----------|--------|---------|--------|
| Button | `h-9` (36px) | `rounded-full` | inset |
| Input | `h-9` (36px) | `rounded-md` | `shadow-xs` |
| Card | auto | `rounded-lg` | `shadow-sm` |
| Badge | auto | `rounded-full` | none |
| Skeleton | varies | `rounded-md` | none |

---

*Last Updated: December 30, 2025 | Version: 2.0*
