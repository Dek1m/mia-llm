"""Живой trace run в Redis — видно с другого воркера, не ждёт commit SQL."""
from __future__ import annotations

import json
import os
from typing import Any

__all__ = [
    "put_trace",
    "get_trace",
    "clear_trace",
    "add_chat_tokens",
    "get_chat_tokens",
]

_TTL = 900
_PREFIX = "llm:trace:"
_CHAT_PREFIX = "llm:chat_stats:"
_CHAT_TTL = 7 * 24 * 3600


def _key(session_id: str) -> str:
    return f"{_PREFIX}{session_id}"


def _chat_key(session_id: str) -> str:
    return f"{_CHAT_PREFIX}{session_id}"


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


def add_chat_tokens(session_id: str, tokens_in: int, tokens_out: int, client: Any | None = None) -> None:
    """Копилка токенов чата: живёт неделю, переживает перезагрузку страницы."""
    if not session_id:
        return
    bus = client or _connect()
    key = _chat_key(session_id)
    pipe = bus.pipeline()
    pipe.hincrby(key, "in", int(tokens_in or 0))
    pipe.hincrby(key, "out", int(tokens_out or 0))
    pipe.expire(key, _CHAT_TTL)
    pipe.execute()


def get_chat_tokens(session_id: str, client: Any | None = None) -> dict[str, int]:
    if not session_id:
        return {"in": 0, "out": 0}
    bus = client or _connect()
    raw = bus.hgetall(_chat_key(session_id)) or {}
    return {"in": int(raw.get("in") or 0), "out": int(raw.get("out") or 0)}
