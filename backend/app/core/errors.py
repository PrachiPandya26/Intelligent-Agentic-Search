from typing import Any, Dict, Optional

from fastapi import Request, status
from fastapi.responses import JSONResponse



class ragchatException(Exception):
    """Base exception for all domain and service errors in ragchat."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ScraperException(ragchatException):
    """Raised when website scraping or extraction fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, details=details)


class PaywallDetectedException(ScraperException):
    """Raised when a hard paywall prevents full document retrieval."""

    def __init__(self, message: str = "A hard paywall blocked full content extraction.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details=details)


class RateLimitExceededException(ragchatException):
    """Raised when all LLM providers or models have exceeded their rate limits."""

    def __init__(self, message: str = "Rate limit exceeded across all available model providers.", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_429_TOO_MANY_REQUESTS, details=details)


class LLMProviderException(ragchatException):
    """Raised when an LLM provider encounters an unrecoverable failure."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_502_BAD_GATEWAY, details=details)


class VectorStoreException(ragchatException):
    """Raised when embedding generation or vector indexing fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, details=details)


class ValidationException(ragchatException):
    """Raised for input parameter or URL validation errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST, details=details)


async def ragchat_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Standardized exception handler for custom ragchat domain errors."""
    status_code = exc.status_code if isinstance(exc, ragchatException) else status.HTTP_400_BAD_REQUEST
    message = exc.message if isinstance(exc, ragchatException) else str(exc)
    details = exc.details if isinstance(exc, ragchatException) else None
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error_type": exc.__class__.__name__,
            "message": message,
            "details": details,
            "path": request.url.path,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for unhandled server exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": "InternalServerError",
            "message": "An unexpected internal server error occurred.",
            "details": {"error": str(exc)},
            "path": request.url.path,
        },
    )
