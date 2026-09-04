"""LLM loop: pipeline assemble → pack → chat. Тулы — после homes."""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from .middleware import TurnCtx, after_run, before_run, run_phase

__all__ = ["assemble_window", "run_loop", "parse_usage"]

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


async def run_loop(
    ctx: TurnCtx,
    steps: list[dict[str, Any]],
    *,
    system_prompt: str | None,
    chat: ChatFn,
    model: str | None = None,
) -> dict[str, Any]:
    ctx = before_run(ctx)
    ctx = await run_phase(ctx, steps, "gather")
    ctx = await run_phase(ctx, steps, "transform")
    ctx.window = assemble_window(ctx, system_prompt)
    result = await chat(messages=ctx.window, model=model)
    usage = parse_usage(result.get("usage") if isinstance(result, dict) else None)
    ctx.usage = usage
    episode = {
        "status": "success",
        "content": (result or {}).get("content") if isinstance(result, dict) else None,
        "model": (result or {}).get("model") if isinstance(result, dict) else None,
        **usage,
    }
    after_run(ctx, episode)
    return episode
