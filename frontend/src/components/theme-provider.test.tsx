import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ThemeProvider, ThemeToggle } from "./theme-provider";
import { THEME_KEY, themeBootstrap } from "@/lib/theme";

beforeEach(() => { localStorage.clear(); document.documentElement.dataset.theme = "dark"; });
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("shared theme", () => {
  it("keeps the saved theme across provider remounts and synchronizes controls", () => {
    const view = render(<ThemeProvider><ThemeToggle /><ThemeToggle showLabel /></ThemeProvider>);
    fireEvent.click(screen.getAllByRole("button", { name: "Light mode" })[0]);
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(localStorage.getItem(THEME_KEY)).toBe("true");
    expect(screen.getAllByRole("button", { name: "Dark mode" })).toHaveLength(2);
    view.unmount();
    render(<ThemeProvider><ThemeToggle /></ThemeProvider>);
    expect(screen.getByRole("button", { name: "Dark mode" })).toBeInTheDocument();
  });

  it("responds to preference changes from another tab", () => {
    render(<ThemeProvider><ThemeToggle /></ThemeProvider>);
    act(() => { localStorage.setItem(THEME_KEY, "true"); window.dispatchEvent(new StorageEvent("storage", { key: THEME_KEY })); });
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(screen.getByRole("button", { name: "Dark mode" })).toBeInTheDocument();
  });

  it("remains usable when browser storage is blocked, including bootstrap", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("Storage denied"); });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("Storage denied"); });
    expect(() => Function(themeBootstrap)()).not.toThrow();
    render(<ThemeProvider><ThemeToggle /></ThemeProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Light mode" }));
    expect(document.documentElement.dataset.theme).toBe("light");
    fireEvent.click(screen.getByRole("button", { name: "Dark mode" }));
    expect(document.documentElement.dataset.theme).toBe("dark");
  });
});
