import { expect, it } from "vitest";
import { ApiError } from "@/lib/api-client";
import { evaluationRequestError } from "./request-error";

it("shows correlation IDs and safe retry guidance for uncertain dispatches", () => {
  const result = evaluationRequestError(new ApiError("Server error", 500, "request-123"), true);
  expect(result).toContain("Request ID: request-123");
  expect(result).toContain("Check Run history");
  expect(result).toContain("unchanged settings");
});

it("does not suggest re-dispatching validation failures or read errors", () => {
  expect(evaluationRequestError(new ApiError("Invalid policy", 422, "request-123"), true)).not.toContain("retrying");
  expect(evaluationRequestError(new ApiError("Server error", 500))).not.toContain("Run history");
  expect(evaluationRequestError(new Error("Response lost"), true)).toContain("Run history");
});
