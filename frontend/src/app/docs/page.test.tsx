import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import DocsPage from "./page";
vi.mock("@/components/public-header", () => ({PublicHeader: () => <header>LLMForge</header>}));
vi.mock("@/components/public-footer", () => ({PublicFooter: () => <footer/>}));
vi.mock("@/components/deployment-info", () => ({DeploymentInfo: () => <div>Deployment identity</div>}));
afterEach(cleanup);
it("documents local SDK execution with no hosted dataset authoring", () => {
  render(<DocsPage/>);
  expect(screen.getByText(/LLMFORGE_URL=https:/)).toHaveTextContent("LLMFORGE_URL=http://localhost:8000/api/v1");
  expect(screen.getByText(/Hosted dataset authoring is retired/)).toBeInTheDocument();
  expect(screen.getByLabelText("Local SDK example")).toHaveTextContent("forge.evaluate_local");
  expect(screen.getByLabelText("Local SDK example")).toHaveTextContent("scoring_version=");
  expect(screen.queryByText(/Applying an import does not save it/)).not.toBeInTheDocument();
});
