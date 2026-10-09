import { ApiError } from "@/lib/api-client";

export function evaluationRequestError(failure: unknown, dispatch = false): string {
  const message = failure instanceof Error ? failure.message : "Request failed";
  const identifier = failure instanceof ApiError && failure.requestId
    ? ` Request ID: ${failure.requestId}.` : "";
  const uncertain = dispatch && (!(failure instanceof ApiError) || failure.statusCode >= 500 || failure.statusCode === 0);
  return message + identifier + (uncertain
    ? " Check Run history before retrying. Retrying unchanged settings reuses this request to avoid duplicate runs." : "");
}
