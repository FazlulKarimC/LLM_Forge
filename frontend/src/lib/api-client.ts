import * as Sentry from "@sentry/nextjs";

export class ApiError extends Error {
    public statusCode: number;
    public requestId?: string;
    public details?: Record<string, unknown>[];
    public retryAfter?: number;

    constructor(message: string, statusCode: number, requestId?: string, details?: Record<string, unknown>[], retryAfter?: number) {
        super(message);
        this.name = "ApiError";
        this.statusCode = statusCode;
        this.requestId = requestId;
        this.details = details;
        this.retryAfter = retryAfter;
    }
}

export type FetchWithHandlingOptions = {
    timeoutMs?: number;
    maxRetries?: number;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

if (process.env.NODE_ENV === "production" && !process.env.NEXT_PUBLIC_API_URL) {
    console.error(
        "[LlmForge] NEXT_PUBLIC_API_URL is not set! " +
        "All API calls will fall back to http://localhost:8000 which will fail in production. " +
        "Set NEXT_PUBLIC_API_URL in your Vercel environment variables."
    );
}

export const SAFE_METHOD_TIMEOUT_MS = 10000;
export const UNSAFE_METHOD_TIMEOUT_MS = 15000;
export const SAFE_METHOD_MAX_RETRIES = 1;
const UNSAFE_METHOD_MAX_RETRIES = 0;
const RETRY_BACKOFF_BASE_MS = 2000;
const BACKGROUND_JOB_POLL_INTERVAL_MS = 2500;
const BACKGROUND_JOB_TIMEOUT_MS = 5 * 60_000;

type BackgroundJobStatus = "queued" | "running" | "completed" | "failed";

interface BackgroundJob<T> {
    job_id: string;
    kind: string;
    status: BackgroundJobStatus;
    created_at: string;
    updated_at: string;
    metadata?: Record<string, unknown>;
    result?: T;
    error?: string;
}

function isSafeRetryMethod(method?: string): boolean {
    const normalizedMethod = (method || "GET").toUpperCase();
    return normalizedMethod === "GET" || normalizedMethod === "HEAD";
}

function getAbortError(): DOMException {
    return new DOMException("The operation was aborted.", "AbortError");
}

function sleep(ms: number, signal?: AbortSignal | null): Promise<void> {
    return new Promise((resolve, reject) => {
        const abortSignal = signal ?? undefined;

        if (abortSignal?.aborted) {
            reject(abortSignal.reason ?? getAbortError());
            return;
        }

        const onAbort = () => {
            clearTimeout(timeout);
            abortSignal?.removeEventListener("abort", onAbort);
            reject(abortSignal?.reason ?? getAbortError());
        };

        const timeout = setTimeout(() => {
            abortSignal?.removeEventListener("abort", onAbort);
            resolve();
        }, ms);

        abortSignal?.addEventListener("abort", onAbort, { once: true });
    });
}

function combineAbortSignals(timeoutSignal: AbortSignal, externalSignal?: AbortSignal | null): AbortSignal {
    if (!externalSignal) {
        return timeoutSignal;
    }

    const abortSignalWithAny = AbortSignal as typeof AbortSignal & {
        any?: (signals: AbortSignal[]) => AbortSignal;
    };

    if (typeof abortSignalWithAny.any === "function") {
        return abortSignalWithAny.any([timeoutSignal, externalSignal]);
    }

    const controller = new AbortController();
    const abortFrom = (signal: AbortSignal) => {
        if (!controller.signal.aborted) {
            controller.abort(signal.reason);
        }
    };

    if (timeoutSignal.aborted) abortFrom(timeoutSignal);
    if (externalSignal.aborted) abortFrom(externalSignal);

    timeoutSignal.addEventListener("abort", () => abortFrom(timeoutSignal), { once: true });
    externalSignal.addEventListener("abort", () => abortFrom(externalSignal), { once: true });

    return controller.signal;
}

async function buildApiError(response: Response): Promise<ApiError> {
    let errorMessage = `API Error: ${response.status}`;
    let requestId = response.headers.get("X-Request-ID") || undefined;
    let details: Record<string, unknown>[] | undefined;
    let retryAfter: number | undefined;

    const retryAfterHeader = response.headers.get("Retry-After");
    if (retryAfterHeader) {
        const parsedRetryAfter = Number(retryAfterHeader);
        if (!Number.isNaN(parsedRetryAfter)) {
            retryAfter = parsedRetryAfter;
        }
    }

    const contentType = response.headers.get("Content-Type") || "";
    try {
        if (contentType.includes("application/json")) {
            const errorData = await response.json();
            errorMessage = errorData.message || errorData.detail || errorMessage;
            requestId = errorData.request_id || requestId;
            details = errorData.details;
            if (response.status === 429 && errorData.retry_after) {
                retryAfter = errorData.retry_after;
            }
        } else {
            const errorText = (await response.text()).trim();
            if (errorText) {
                errorMessage = errorText;
            }
        }
    } catch {
        // Fall back to status code only.
    }

    return new ApiError(errorMessage, response.status, requestId, details, retryAfter);
}

export async function fetchWithHandling(
    url: string,
    options: RequestInit = {},
    config: FetchWithHandlingOptions = {},
): Promise<Response> {
    const safeMethod = isSafeRetryMethod(options.method);
    const {
        timeoutMs = safeMethod ? SAFE_METHOD_TIMEOUT_MS : UNSAFE_METHOD_TIMEOUT_MS,
        maxRetries = safeMethod ? SAFE_METHOD_MAX_RETRIES : UNSAFE_METHOD_MAX_RETRIES,
    } = config;
    const externalSignal = options.signal;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        const start = Date.now();
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        const combinedSignal = combineAbortSignals(controller.signal, externalSignal);

        try {
            if (externalSignal?.aborted) {
                throw externalSignal.reason ?? getAbortError();
            }

            const headers = new Headers(options.headers);
            const hasBody = options.body !== undefined && options.body !== null;
            if (hasBody && !headers.has("Content-Type") && !(options.body instanceof FormData)) {
                headers.set("Content-Type", "application/json");
            }

            const response = await fetch(url, {
                ...options,
                signal: combinedSignal,
                headers,
            });

            clearTimeout(timeout);

            if (!response.ok) {
                const apiError = await buildApiError(response);
                if (safeMethod && response.status >= 500 && attempt < maxRetries) {
                    await sleep(RETRY_BACKOFF_BASE_MS * Math.pow(2, attempt), externalSignal);
                    continue;
                }
                throw apiError;
            }

            return response;
        } catch (err) {
            clearTimeout(timeout);

            if (err instanceof ApiError) {
                Sentry.addBreadcrumb({
                    category: "api",
                    message: `${options.method ?? "GET"} ${url}`,
                    level: "error",
                    data: {
                        status: err.statusCode,
                        duration_ms: Date.now() - start,
                        request_id: err.requestId,
                    },
                });
                throw err;
            }

            if (externalSignal?.aborted && !controller.signal.aborted) {
                throw err;
            }

            if (safeMethod && attempt < maxRetries) {
                await sleep(RETRY_BACKOFF_BASE_MS * Math.pow(2, attempt), externalSignal);
                continue;
            }

            if (err instanceof DOMException && err.name === "AbortError") {
                if (controller.signal.aborted && !externalSignal?.aborted) {
                    throw new ApiError("Request timed out. The backend may still be waking up on Hugging Face Spaces.", 408);
                }
                throw err;
            }

            throw new ApiError(err instanceof Error ? err.message : "Network error", 0);
        }
    }

    throw new ApiError("Unexpected retry exhaustion", 500);
}

export async function fetchAPI<T>(
    endpoint: string,
    options: RequestInit = {},
    config: FetchWithHandlingOptions = {},
): Promise<T> {
    const response = await fetchWithHandling(`${API_BASE_URL}${endpoint}`, options, config);

    if (response.status === 204) {
        return undefined as unknown as T;
    }

    return response.json();
}

export function getApiBaseUrl(): string {
    return API_BASE_URL;
}

export function getApiRootUrl(): string {
    return API_BASE_URL.replace(/\/api\/v1\/?$/, "") || "http://localhost:8000";
}

export async function waitForBackgroundJob<T>(jobId: string, signal?: AbortSignal | null): Promise<T> {
    const deadline = Date.now() + BACKGROUND_JOB_TIMEOUT_MS;

    while (Date.now() < deadline) {
        const response = await fetchWithHandling(
            `${API_BASE_URL}/results/jobs/${jobId}`,
            { signal },
            { timeoutMs: SAFE_METHOD_TIMEOUT_MS, maxRetries: SAFE_METHOD_MAX_RETRIES },
        );
        const job = await response.json() as BackgroundJob<T>;

        if (job.status === "completed" && job.result !== undefined) {
            return job.result;
        }

        if (job.status === "failed") {
            throw new ApiError(job.error || "Background job failed", 500);
        }

        await sleep(BACKGROUND_JOB_POLL_INTERVAL_MS, signal);
    }

    throw new ApiError("Background job timed out while polling for completion.", 408);
}
