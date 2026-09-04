from __future__ import annotations

import pytest

from modules.llm.loop import assemble_window, compose_system_prompt, parse_usage, run_loop
from modules.llm.middleware import DEFAULT_PIPELINE, TurnCtx, list_catalog, run_phase


def test_catalog_has_main_algorithm_steps() -> None:
    ids = {item["id"] for item in list_catalog()}
    assert ids >= {"chat_branch", "compress_window", "memory_recall"}
    assert DEFAULT_PIPELINE["name"] == "Main Algorithm"


def test_parse_usage_cached() -> None:
    got = parse_usage(
        {"prompt_tokens": 10, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 3}},
    )
    assert got["tokens_in"] == 10
    assert got["tokens_out"] == 4
    assert got["cache_tokens"] == 3
    assert got["cache_hits"] == 1


def test_compose_system_prompt_agent_and_user() -> None:
    text = compose_system_prompt(
        "You are Build",
        {"username": "sereja", "email": "a@b.c", "user_prompt": "Be brief", "nickname": ""},
        agent_name="build",
    )
    assert text is not None
    assert text.startswith("You are Build")
    assert "username: sereja" in text
    assert "email: a@b.c" in text
    assert "user_prompt: Be brief" in text
    assert "nickname:" not in text


def test_compose_system_prompt_fallback_name() -> None:
    text = compose_system_prompt(None, None, agent_name="Athena", agent_description="Lead")
    assert text == "You are Athena. Lead"


def test_assemble_window_system_and_history() -> None:
    ctx = TurnCtx(
        transcript=[{"role": "user", "content": "hi"}],
        retrieved=[],
        query="hi",
    )
    messages = assemble_window(ctx, "You are Build")
    assert messages[0] == {"role": "system", "content": "You are Build"}
    assert messages[-1]["role"] == "user"


def test_get_model_lookup_exists() -> None:
    from modules.llm.repository import LLMRepository

    assert hasattr(LLMRepository, "get_model")


@pytest.mark.asyncio
async def test_run_loop_echoes_chat() -> None:
    async def chat(*, messages, model=None):
        assert messages
        return {
            "content": "ok",
            "model": "fake",
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }

    ctx = TurnCtx(query="hi", transcript=[{"role": "user", "content": "hi"}])
    episode = await run_loop(
        ctx,
        DEFAULT_PIPELINE["steps"],
        system_prompt="sys",
        chat=chat,
        model=None,
    )
    assert episode["content"] == "ok"
    assert episode["tokens_in"] == 2


@pytest.mark.asyncio
async def test_compress_keeps_budget() -> None:
    ctx = TurnCtx(budget_chars=8, query="x")
    from modules.llm.middleware import Retrieved

    ctx.retrieved = [
        Retrieved(text="abcdefghij", source="memory_recall"),
        Retrieved(text="zz", source="memory_recall"),
    ]
    await run_phase(
        ctx,
        [{"ord": 1, "middleware_id": "compress_window", "phase": "transform", "enabled": True, "config": {}}],
        "transform",
    )
    assert ctx.retrieved
    assert sum(len(item.text) for item in ctx.retrieved) <= 8 or len(ctx.retrieved) == 1
