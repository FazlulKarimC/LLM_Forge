import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ServerCapabilities } from "./server-capabilities";
import { fetchAPI } from "@/lib/api-client";

vi.mock("@/lib/api-client", () => ({ fetchAPI: vi.fn() }));
afterEach(() => { cleanup(); vi.resetAllMocks(); });

describe("deployed evaluation providers", () => {
  it("displays server choices and the local Gemini submission workflow", async () => {
    vi.mocked(fetchAPI).mockResolvedValue({ generation_providers: ["mock", "groq"], judge_providers: ["groq"], external_submissions: true });
    render(<ServerCapabilities />);
    expect(await screen.findByText(/Server generation: Demo \(mock\), Groq. Judges: Groq./)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "External results guide" })).toHaveAttribute("href", "/docs#external");
    expect(screen.getByText(/require your provider credentials/)).toBeInTheDocument();
  });
  it("keeps a useful state when the backend predates capabilities discovery", async () => {
    vi.mocked(fetchAPI).mockRejectedValue(new Error("Not found"));
    render(<ServerCapabilities />);
    expect(await screen.findByText(/Server capabilities are unavailable/)).toBeInTheDocument();
    expect(screen.queryByText(/Server generation:/)).not.toBeInTheDocument();
  });
});
