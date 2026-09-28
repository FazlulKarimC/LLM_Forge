import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { useUnsavedChanges } from "./use-unsaved-changes";

function Draft({ dirty }: { dirty: boolean }) {
  useUnsavedChanges(dirty);
  return <a href="/evaluations">Evaluations</a>;
}

afterEach(() => vi.restoreAllMocks());

it("blocks internal navigation and project switching when a draft is unsaved", () => {
  window.history.replaceState(null, "", "/datasets");
  vi.spyOn(window, "confirm").mockReturnValue(false);
  const view = render(<Draft dirty />);
  const link = screen.getByRole("link", { name: "Evaluations" });
  expect(fireEvent.click(link)).toBe(false);
  expect(window.dispatchEvent(new Event("llmforge:before-workspace-switch", { cancelable: true }))).toBe(false);
  expect(window.confirm).toHaveBeenCalledTimes(2);
  view.rerender(<Draft dirty={false} />);
  expect(window.dispatchEvent(new Event("llmforge:before-workspace-switch", { cancelable: true }))).toBe(true);
});
