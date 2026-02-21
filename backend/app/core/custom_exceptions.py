from typing import Any, Dict, Optional

class AppException(Exception):
    """
    Base exception for all custom application errors.
    """
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.context = context or {}

class ResourceNotFoundException(AppException):
    """
    Raised when a requested resource (e.g., database record) is not found.
    """
    def __init__(self, resource_type: str, resource_id: Any):
        super().__init__(
            message=f"{resource_type} with id {resource_id} not found",
            status_code=404,
            context={"resource_type": resource_type, "resource_id": str(resource_id)},
        )

class ValidationException(AppException):
    """
    Raised when business logic validation fails (different from Pydantic schema validation).
    """
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=400,
            context=context,
        )

class UnauthorizedException(AppException):
    """
    Raised when authentication or authorization fails.
    """
    def __init__(self, message: str = "Unauthorized access"):
        super().__init__(
            message=message,
            status_code=401,
        )
