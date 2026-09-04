"""Живой trace run в Redis — видно с другого воркера, не ждёт commit SQL."""
from __future__ import annotations

import json
import os
from typing import Any

__all__ = ["put_trace", "get_trace", "clear_trace"]

_TTL = 900
_PREFIX = "llm:trace:"


def _key(session_id: str) -> str:
    return f"{_PREFIX}{session_id}"


def _connect() -> Any:
    import redis

    host = os.environ.get("REDIS_HOST") or os.environ.get("WORKER_REDIS_HOST") or "127.0.0.1"
    port = int(os.environ.get("REDIS_PORT") or os.environ.get("WORKER_REDIS_PORT") or 6379)
    return redis.Redis(host=host, port=port, decode_responses=True)


def put_trace(session_id: str, trace: dict[str, Any], client: Any | None = None) -> None:
    if not session_id:
        return
    bus = client or _connect()
    bus.set(_key(session_id), json.dumps(trace), ex=_TTL)


def get_trace(session_id: str, client: Any | None = None) -> dict[str, Any] | None:
    if not session_id:
        return None
    bus = client or _connect()
    raw = bus.get(_key(session_id))
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def clear_trace(session_id: str, client: Any | None = None) -> None:
    if not session_id:
        return
    bus = client or _connect()
    bus.delete(_key(session_id))
