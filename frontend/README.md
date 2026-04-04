# LlmForge - Frontend Interface

> The interactive web dashboard and experiment manager for the LlmForge platform.

This directory contains the frontend web application built to interface with the FastAPI backend. It allows users to create experiments, monitor long-running background reasoning tasks in real-time, and analyze detailed execution metrics side-by-side.

---

## 🛠️ Technology Stack

- **Framework:** [Next.js 16](https://nextjs.org/) (App Router)
- **UI Library:** [React 19](https://react.dev/)
- **Styling:** [Tailwind CSS v4](https://tailwindcss.com/)
- **Components:** [shadcn/ui](https://ui.shadcn.com/)
- **Data Fetching:** [TanStack React Query](https://tanstack.com/query/latest) (progressively loaded fetching & state)
- **Observability:** Sentry (Next.js Edge/Server/Client tracking)
- **Icons:** [Lucide React](https://lucide.dev/)

---

## 🎨 Design System

All frontend code strictly adheres to the unified 4-color palette and typography rules defined in the root-level `DESIGN_SYSTEM.md` document. 

*If you are an LLM agent writing code or a developer adding a new component, you must consult `../DESIGN_SYSTEM.md` before proceeding.*

---

## 📁 Project Structure

```text
src/
├── app/                  # Next.js App Router Pages
│   ├── (app)/            # Authenticated/Main App Routes Group
│   │   ├── dashboard/    # Operational Dashboard
│   │   └── experiments/  # Experiment views (List, Detail, Compare)
│   ├── globals.css       # Tailwind configuration & core variables
│   ├── layout.tsx        # Root layout, font definitions, Navbars
│   ├── providers.tsx     # Context Providers (Query, Sentry, Theme)
│   └── page.tsx          # Landing / Home Page
├── components/           # Reusable React components
│   └── ui/               # shadcn/ui & domain components (RoutingPanel, RegressionPanel)
└── lib/                  # Utilities
    ├── api.ts            # Typed API client routing to NEXT_PUBLIC_API_URL
    └── utils.ts          # clsx + tailwind-merge utilities
```

---

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- The LlmForge FastAPI backend **must** be running (default `http://localhost:8000`), as the frontend aggressively calls the API to poll run statuses.

### Installation

Navigate to this directory and install dependencies:

```bash
npm install
```

### Development Server

Start the interactive development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the application. The app features hot-reloading for rapid UI development.

### Production Build

To create an optimized production build:

```bash
npm run build
npm run start
```

---

## 📡 API Routing Note

By default, the `api.ts` client expects the backend API to be available on `http://localhost:8000`. This can be configured by setting `NEXT_PUBLIC_API_URL` to your production backend domain in the `.env` file or hosting environment.
