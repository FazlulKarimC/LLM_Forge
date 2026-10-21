"""Bound local-summary request bytes before FastAPI parses JSON or dependencies."""

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute

LOCAL_REPORT_MAX_BYTES = 64_000


class LocalReportRequest(Request):
    async def body(self) -> bytes:
        if not hasattr(self, "_body"):
            body = bytearray()
            async for chunk in self.stream():
                if len(body) + len(chunk) > LOCAL_REPORT_MAX_BYTES:
                    raise HTTPException(413, "Local report request bodies are limited to 64,000 bytes")
                body.extend(chunk)
            self._body = bytes(body)
        return self._body


class LocalReportRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def bounded_handler(request: Request):
            return await handler(LocalReportRequest(request.scope, request.receive))

        return bounded_handler
