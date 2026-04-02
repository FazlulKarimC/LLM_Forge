"use client";

import { QueryClient, QueryClientProvider, QueryCache, MutationCache } from "@tanstack/react-query";
import * as Sentry from "@sentry/nextjs";
import { useState, type ReactNode } from "react";

export function Providers({ children }: { children: ReactNode }) {
    const [queryClient] = useState(
        () =>
            new QueryClient({
                queryCache: new QueryCache({
                    onError: (error, query) =>
                        Sentry.captureException(error, {
                            extra: { queryKey: query.queryKey },
                        }),
                }),
                mutationCache: new MutationCache({
                    onError: (error) => Sentry.captureException(error),
                }),
                defaultOptions: {
                    queries: {
                        staleTime: 60_000,           // 1 minute
                        retry: 1,                    // down from default 3
                        retryDelay: 2000,            // fixed 2s backoff
                        refetchOnWindowFocus: false,  // already set
                        refetchOnReconnect: false,    // prevents burst on reconnect
                    },
                },
            })
    );

    return (
        <QueryClientProvider client={queryClient}>
            {children}
        </QueryClientProvider>
    );
}
