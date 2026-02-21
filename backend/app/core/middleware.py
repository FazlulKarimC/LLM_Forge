import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware that generates a unique request_id for each incoming request,
    attaches it to the request state, and includes it in the response headers.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Attach to the request state so route handlers and exception handlers can use it
        request.state.request_id = request_id
        
        # Process the request
        response = await call_next(request)
        
        # Include the request ID in the response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
