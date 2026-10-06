import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import LandingPage from "@/app/page";
import { ThemeProvider } from "@/components/theme-provider";

beforeEach(() => { localStorage.clear(); document.documentElement.dataset.theme = "dark"; });
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("public landing walkthrough", () => {
  it("explores all previews with keyboard focus and makes no application API calls", () => {
    const fetch = vi.spyOn(globalThis, "fetch");
    render(<ThemeProvider><LandingPage /></ThemeProvider>);
    const benchmarks = screen.getByRole("tab", { name: "Benchmarks" });
    const prompts = screen.getByRole("tab", { name: "Prompts" });
    const evaluations = screen.getByRole("tab", { name: "Evaluations" });
    expect(benchmarks).toHaveAttribute("aria-selected", "true");
    benchmarks.focus();
    fireEvent.keyDown(benchmarks, { key: "ArrowRight" });
    expect(prompts).toHaveFocus();
    expect(screen.getByRole("tabpanel")).toHaveAttribute("aria-labelledby", prompts.id);
    fireEvent.keyDown(prompts, { key: "End" });
    expect(evaluations).toHaveFocus();
    fireEvent.keyDown(evaluations, { key: "ArrowRight" });
    expect(benchmarks).toHaveFocus();
    fireEvent.keyDown(benchmarks, { key: "ArrowLeft" });
    expect(evaluations).toHaveFocus();
    fireEvent.keyDown(evaluations, { key: "Home" });
    expect(benchmarks).toHaveFocus();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("offers a public preview and explains authenticated demo entry", () => {
    render(<ThemeProvider><LandingPage /></ThemeProvider>);
    expect(screen.getByRole("link", { name: "View walkthrough" })).toHaveAttribute("href", "#walkthrough");
    expect(screen.getByRole("link", { name: "View GitHub" })).toHaveAttribute("href", "https://github.com/FazlulKarimC/LLM_Forge");
    expect(screen.getByText(/Sign in to your workspace/)).toHaveTextContent("Create demo examples");
    expect(screen.getByRole("tabpanel")).toHaveTextContent("not claims about model performance");
  });

  it("closes mobile navigation on selection and Escape", () => {
    render(<ThemeProvider><LandingPage /></ThemeProvider>);
    const menu = screen.getByLabelText("Navigation").closest("details")!;
    menu.open = true;
    const docsLink = within(menu).getByRole("link", { name: "Docs" });
    docsLink.addEventListener("click", (event) => event.preventDefault());
    fireEvent.click(docsLink);
    expect(menu.open).toBe(false);
    menu.open = true;
    fireEvent.keyDown(within(menu).getByRole("link", { name: "Product" }), { key: "Escape" });
    expect(menu.open).toBe(false);
    expect(screen.getByLabelText("Navigation")).toHaveFocus();
  });
});
