import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { AnimatedNumber, MetricBar } from "./primitives";

vi.mock("framer-motion", () => ({ useReducedMotion: () => true }));

it("renders final metrics without scheduling frames when reduced motion is preferred", () => {
  const frames = vi.spyOn(window, "requestAnimationFrame");
  const { container, rerender } = render(<><AnimatedNumber value={42} suffix="%" /><MetricBar value={42} /></>);
  expect(screen.getByText("42%")).toBeInTheDocument();
  expect(container.querySelector('[style*="width"]')).toHaveStyle({ width: "42%", transition: "none" });
  rerender(<><AnimatedNumber value={73} suffix="%" /><MetricBar value={73} /></>);
  expect(screen.getByText("73%")).toBeInTheDocument();
  expect(frames).not.toHaveBeenCalled();
  frames.mockRestore();
});
