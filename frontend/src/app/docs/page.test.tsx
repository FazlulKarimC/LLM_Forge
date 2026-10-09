import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import DocsPage from "./page";
vi.mock("@/components/public-header", () => ({ PublicHeader: () => <header>LLMForge</header> }));
vi.mock("@/components/public-footer", () => ({ PublicFooter: () => <footer /> }));
vi.mock("@/components/server-capabilities", () => ({ ServerCapabilities: () => <div>Deployed providers</div> }));
vi.mock("@/components/deployment-info", () => ({ DeploymentInfo: () => <div>Deployment identity</div> }));
afterEach(cleanup);

it("provides a complete hosted consumer target and authoring journey", () => {
  render(<DocsPage />);
  expect(screen.getByText(/LLMFORGE_URL=https:\/\/faz-ai-llm-forge.hf.space\/api\/v1/)).toHaveTextContent("LLMFORGE_URL=http://localhost:8000/api/v1");
  expect(screen.getByText(/Applying an import does not save it/)).toBeInTheDocument();
  expect(screen.getByText(/numeric deltas are disabled even when submitted check names match/)).toBeInTheDocument();
  expect(screen.getByText(/Direct Gemini server generation is currently unavailable/)).toBeInTheDocument();
});
