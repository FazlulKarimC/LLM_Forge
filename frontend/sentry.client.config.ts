import * as Sentry from "@sentry/nextjs";

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,

  // Error-first: no performance tracing, no session replay
  tracesSampleRate: 0,
  replaysSessionSampleRate: 0,
  replaysOnErrorSampleRate: 0,

  // Scrub LLM content from events before sending to Sentry
  beforeSend(event) {
    if (event.extra) {
      delete event.extra.prompt;
      delete event.extra.raw_output;
      delete event.extra.expected_output;
    }
    return event;
  },
});
