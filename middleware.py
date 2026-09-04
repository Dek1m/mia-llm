"""Каталог middleware сборки окна. Код, не загрузка с админки."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Retrieved",
    "TurnCtx",
    "MIDDLEWARE",
    "DEFAULT_PIPELINE",
    "list_catalog",
    "run_phase",
    "before_run",
    "after_run",
]


@dataclass
class Retrieved:
    text: str
    source: str
    score: float = 1.0
    id: str = ""


@dataclass
class TurnCtx:
    query: str = ""
    workspace_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    transcript: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[Retrieved] = field(default_factory=list)
    window: list[dict[str, Any]] = field(default_factory=list)
    budget_chars: int = 24_000
    usage: dict[str, int] = field(default_factory=dict)
    cancel: bool = False


async def _workspace_rag(ctx: TurnCtx, _config: dict[str, Any]) -> TurnCtx:
    return ctx


async def _files_catalog(ctx: TurnCtx, _config: dict[str, Any]) -> TurnCtx:
    return ctx


async def _chat_branch(ctx: TurnCtx, config: dict[str, Any]) -> TurnCtx:
    limit = int(config.get("max_messages") or 40)
    for item in ctx.transcript[-limit:]:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            ctx.retrieved.append(
                Retrieved(text=content, source="chat_branch", id=str(item.get("id") or "")),
            )
    return ctx


async def _memory_recall(ctx: TurnCtx, _config: dict[str, Any]) -> TurnCtx:
    """MemoryPort. Default adapter — пусто (selti позже)."""
    return ctx


async def _compress_window(ctx: TurnCtx, _config: dict[str, Any]) -> TurnCtx:
    used = 0
    kept: list[Retrieved] = []
    for item in reversed(ctx.retrieved):
        size = len(item.text)
        if used + size > ctx.budget_chars and kept:
            break
        kept.append(item)
        used += size
    kept.reverse()
    ctx.retrieved = kept
    return ctx


MIDDLEWARE: dict[str, dict[str, Any]] = {
    "workspace_rag": {
        "id": "workspace_rag",
        "kind": "source",
        "phase": "gather",
        "run": _workspace_rag,
    },
    "files_catalog": {
        "id": "files_catalog",
        "kind": "source",
        "phase": "gather",
        "run": _files_catalog,
    },
    "chat_branch": {
        "id": "chat_branch",
        "kind": "source",
        "phase": "gather",
        "run": _chat_branch,
    },
    "memory_recall": {
        "id": "memory_recall",
        "kind": "source",
        "phase": "gather",
        "run": _memory_recall,
    },
    "compress_window": {
        "id": "compress_window",
        "kind": "transform",
        "phase": "transform",
        "run": _compress_window,
    },
}

DEFAULT_PIPELINE: dict[str, Any] = {
    "name": "Main Algorithm",
    "slug": "main-algorithm",
    "purpose": "Default context gather and LLM loop",
    "caps": {"max_turns": 8, "budget_chars": 24000},
    "steps": [
        {"ord": 10, "middleware_id": "workspace_rag", "phase": "gather", "enabled": True, "config": {}},
        {"ord": 20, "middleware_id": "files_catalog", "phase": "gather", "enabled": True, "config": {}},
        {"ord": 30, "middleware_id": "chat_branch", "phase": "gather", "enabled": True, "config": {"max_messages": 40}},
        {"ord": 40, "middleware_id": "memory_recall", "phase": "gather", "enabled": True, "config": {"k": 8}},
        {"ord": 50, "middleware_id": "compress_window", "phase": "transform", "enabled": True, "config": {}},
    ],
}


def list_catalog() -> list[dict[str, Any]]:
    return [
        {"id": item["id"], "kind": item["kind"], "phase": item["phase"]}
        for item in MIDDLEWARE.values()
    ]


async def run_phase(ctx: TurnCtx, steps: list[dict[str, Any]], phase: str) -> TurnCtx:
    selected = [
        step for step in steps
        if step.get("phase") == phase and step.get("enabled", True)
    ]
    # Sources пока пишут в один retrieved — строго по ord, без гонки.
    for step in sorted(selected, key=lambda item: int(item.get("ord") or 0)):
        spec = MIDDLEWARE.get(str(step.get("middleware_id") or ""))
        if spec is None:
            continue
        await spec["run"](ctx, step.get("config") or {})
    return ctx


def before_run(ctx: TurnCtx) -> TurnCtx:
    """Дыра ishtar. Identity."""
    return ctx


def after_run(_ctx: TurnCtx, _episode: dict[str, Any]) -> None:
    """Дыра ishtar. Identity."""
    return None
