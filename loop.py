"""LLM loop: pipeline assemble → pack → chat. Тулы — после homes."""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from prometheus_client import REGISTRY, Counter, Histogram

from .middleware import TurnCtx, after_run, before_run, run_phase

__all__ = [
    "assemble_window",
    "compose_system_prompt",
    "run_loop",
    "parse_usage",
    "mark_pipeline",
    "stages_from_output",
]

_USER_FIELDS = (
    ("username", "username"),
    ("nickname", "nickname"),
    ("first_name", "first_name"),
    ("last_name", "last_name"),
    ("date_of_birth", "date_of_birth"),
    ("email", "email"),
    ("phone", "phone"),
    ("chip_display_mode", "chip_display_mode"),
    ("primary_group_id", "primary_group_id"),
    ("user_prompt", "user_prompt"),
)


def _counter(name: str, documentation: str, labelnames: list[str]) -> Counter:
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing  # type: ignore[return-value]
    return Counter(name, documentation, labelnames)


def _histogram(name: str, documentation: str) -> Histogram:
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing  # type: ignore[return-value]
    return Histogram(name, documentation)


llm_pipeline_total = _counter(
    "llm_pipeline_total",
    "LLM pipeline runs",
    ["status"],
)
llm_pipeline_duration_seconds = _histogram(
    "llm_pipeline_duration_seconds",
    "LLM pipeline duration in seconds",
)


def mark_pipeline(status: str, duration_s: float) -> None:
    llm_pipeline_total.labels(status=status).inc()
    llm_pipeline_duration_seconds.observe(duration_s)

ChatFn = Callable[..., Awaitable[dict[str, Any]]]


def parse_usage(raw: dict[str, Any] | None) -> dict[str, int]:
    data = raw or {}
    details = data.get("prompt_tokens_details") or {}
    cached = (
        data.get("cache_tokens")
        or data.get("cached_tokens")
        or details.get("cached_tokens")
        or data.get("cache_read_input_tokens")
        or 0
    )
    hits = 1 if int(cached or 0) > 0 else 0
    return {
        "tokens_in": int(data.get("prompt_tokens") or 0),
        "tokens_out": int(data.get("completion_tokens") or 0),
        "cache_tokens": int(cached or 0),
        "cache_hits": hits,
    }


def compose_system_prompt(
    agent_prompt: str | None,
    user_profile: dict[str, Any] | None = None,
    *,
    agent_name: str | None = None,
    agent_description: str | None = None,
) -> str | None:
    """System: промпт агента из настроек + поля залогиненного пользователя."""
    parts: list[str] = []
    name = (agent_name or "").strip()
    if name:
        parts.append(
            f"Your name is {name}. Answer only as {name}. "
            "Do not claim to be Claude, ChatGPT, Gemini, Grok, or any other assistant "
            "unless that is your name."
        )
    prompt = (agent_prompt or "").strip()
    if not prompt:
        desc = (agent_description or "").strip()
        if name and desc:
            prompt = f"You are {name}. {desc}"
        elif name:
            prompt = f"You are {name}."
    if prompt:
        parts.append(prompt)
    lines: list[str] = []
    for key, label in _USER_FIELDS:
        value = (user_profile or {}).get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            lines.append(f"{label}: {text}")
    if lines:
        parts.append("User:\n" + "\n".join(lines))
    joined = "\n\n".join(parts).strip()
    return joined or None


def assemble_window(ctx: TurnCtx, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    prompt = (system_prompt or "").strip()
    extra = [
        item.text for item in ctx.retrieved
        if item.source != "chat_branch" and item.text.strip()
    ]
    if extra:
        block = "Context:\n" + "\n---\n".join(extra)
        prompt = f"{prompt}\n\n{block}".strip() if prompt else block
    if prompt:
        messages.append({"role": "system", "content": prompt})
    for item in ctx.transcript:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "")
        if role in {"user", "assistant", "system"} and content:
            messages.append({"role": role, "content": content})
    if not any(item["role"] == "user" for item in messages) and ctx.query:
        messages.append({"role": "user", "content": ctx.query})
    return messages


def stages_from_output(reasoning: str, content: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if reasoning:
        items.append({
            "kind": "reasoning",
            "name": "Reasoning",
            "status": "done" if (content or tools) else "running",
        })
    for tool in tools:
        items.append({
            "kind": "tool",
            "name": str(tool.get("name") or "tool"),
            "args": str(tool.get("args") or ""),
            "status": str(tool.get("status") or "done"),
        })
    if content:
        items.append({"kind": "text", "name": "Answer", "status": "done"})
    return items


async def run_loop(
    ctx: TurnCtx,
    steps: list[dict[str, Any]],
    *,
    system_prompt: str | None,
    chat: ChatFn,
    model: str | None = None,
    on_delta: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    ctx = before_run(ctx)
    ctx = await run_phase(ctx, steps, "gather")
    ctx = await run_phase(ctx, steps, "transform")
    ctx.window = assemble_window(ctx, system_prompt)

    async def _emit(trace: dict[str, Any]) -> None:
        if on_delta is None:
            return
        result = on_delta(trace)
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]

    try:
        result = await chat(messages=ctx.window, model=model, on_delta=on_delta)
    except TypeError:
        result = await chat(messages=ctx.window, model=model)
    usage = parse_usage(result.get("usage") if isinstance(result, dict) else None)
    ctx.usage = usage
    content = str((result or {}).get("content") or "") if isinstance(result, dict) else ""
    reasoning = str((result or {}).get("reasoning") or "") if isinstance(result, dict) else ""
    tools = list((result or {}).get("tools") or []) if isinstance(result, dict) else []
    stages = stages_from_output(reasoning, content, tools)
    await _emit({"content": content, "reasoning": reasoning, "stages": stages})
    episode = {
        "status": "success",
        "content": content or None,
        "reasoning": reasoning or None,
        "stages": stages,
        "model": (result or {}).get("model") if isinstance(result, dict) else None,
        **usage,
    }
    after_run(ctx, episode)
    return episode
