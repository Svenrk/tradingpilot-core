from __future__ import annotations

import contextvars
import logging
import re
from collections.abc import Awaitable, Callable
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-"
)
_SECRET_RE = re.compile(r"(?i)(password|secret|token)=([^\s&]+)")
_BASE_RECORD_FACTORY = logging.getLogRecordFactory()


class SecretRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rendered = record.getMessage()
        record.msg = _SECRET_RE.sub(r"\1=[REDACTED]", rendered)
        record.args = ()
        return True


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))
        token = correlation_id_var.set(correlation_id)
        try:
            response = await call_next(request)
        finally:
            correlation_id_var.reset(token)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s [%(name)s] [cid=%(correlation_id)s] %(message)s",
        )
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in root.handlers:
        if not any(isinstance(filter_, SecretRedactionFilter) for filter_ in handler.filters):
            handler.addFilter(SecretRedactionFilter())
    logging.setLogRecordFactory(_record_factory)


def _record_factory(*args, **kwargs):
    record = _BASE_RECORD_FACTORY(*args, **kwargs)
    record.correlation_id = correlation_id_var.get()
    return record
