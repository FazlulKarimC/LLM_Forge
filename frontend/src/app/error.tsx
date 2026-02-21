'use client';

import { useEffect } from 'react';

// Attempt to parse out ApiError properties if the thrown error was one
// (Next.js sometimes strips custom error classes in production boundaries depending on how it's thrown,
// so we also just check for standard properties if possible).
export default function GlobalError({
    error,
    reset,
}: {
    error: Error & { digest?: string; statusCode?: number; requestId?: string; details?: any };
    reset: () => void;
}) {
    useEffect(() => {
        // Log the error securely to the browser console for developer debugging
        console.error('LlmForge Error Boundary Caught:', error);
        if (error.requestId) {
            console.error('Request ID:', error.requestId);
        }
    }, [error]);

    const statusCode = error.statusCode || 500;
    const isNetworkError = error.message.toLowerCase().includes('failed to fetch');

    return (
        <div className="min-h-[60vh] flex items-center justify-center p-6">
            <div className="max-w-md w-full text-center space-y-6">
                <div className="mx-auto w-16 h-16 bg-red-100 rounded-full flex items-center justify-center">
                    <svg
                        className="w-8 h-8 text-red-600"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        aria-hidden="true"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                        />
                    </svg>
                </div>

                <div>
                    <h2 className="text-2xl font-bold text-gray-900 mb-2">
                        {isNetworkError ? 'Connection Issue' : 'Something went wrong'}
                    </h2>

                    <p className="text-gray-600">
                        {isNetworkError
                            ? "We couldn't connect to the server. Please check your internet connection and try again."
                            : error.message || "An unexpected error occurred while processing your request."}
                    </p>

                    {error.requestId && (
                        <p className="mt-4 text-xs font-mono text-gray-400">
                            Request ID: {error.requestId}
                        </p>
                    )}
                </div>

                {error.details && Array.isArray(error.details) && (
                    <div className="mt-4 text-left bg-red-50 p-4 rounded-md border border-red-100">
                        <ul className="list-disc list-inside text-sm text-red-700 space-y-1">
                            {error.details.map((detail: any, idx: number) => (
                                <li key={idx}>
                                    <span className="font-semibold">{detail.field}:</span> {detail.issue}
                                </li>
                            ))}
                        </ul>
                    </div>
                )}

                <div className="pt-4">
                    <button
                        onClick={
                            // Attempt to recover by trying to re-render the segment
                            () => reset()
                        }
                        className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors"
                    >
                        Try Again
                    </button>
                </div>
            </div>
        </div>
    );
}
