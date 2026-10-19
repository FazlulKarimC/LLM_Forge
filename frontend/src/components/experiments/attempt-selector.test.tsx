import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { expect, it, vi } from "vitest";
import { AttemptSelector } from "./attempt-selector";

const getAttempts = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api", () => ({ getBenchmarkAttempts: getAttempts, exportResults: vi.fn(), purgeBenchmarkAttempt: vi.fn() }));

it("refreshes first-attempt provenance when execution changes status", async () => {
  getAttempts.mockResolvedValueOnce({ current_attempt: 1, total: 0, attempts: [] });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const view = (status: string) => <QueryClientProvider client={client}><AttemptSelector id="smoke" current={1} selected={1} status={status} onSelect={() => {}} /></QueryClientProvider>;
  const rendered = render(view("pending"));
  await waitFor(() => expect(getAttempts).toHaveBeenCalledTimes(1));
  getAttempts.mockResolvedValue({ current_attempt: 1, total: 1, attempts: [{ number: 1, status: "completed", provenance: "recorded", selected: 3 }] });
  rendered.rerender(view("completed"));
  await waitFor(() => expect(screen.getByRole("option", { name: "1 · completed · recorded" })).toBeTruthy());
  expect(getAttempts).toHaveBeenCalledTimes(2);
});
