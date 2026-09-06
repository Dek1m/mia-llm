"""Вендор-адаптер режимов reasoning для OpenAI-совместимых чатов.

Единого стандарта нет: OpenAI/o-серии ждут reasoning_effort, GLM — thinking.type,
Qwen — enable_thinking, OpenRouter — reasoning.effort, DeepSeek выбирает режим
моделью. Здесь один вход (base_url, model_id, effort) — один выход (payload-dict).
"""
from __future__ import annotations

from typing import Any

_EFFORTS = {"low", "medium", "high"}


def _vendor(base_url: str) -> str:
    url = (base_url or "").lower()
    if "z.ai" in url or "bigmodel" in url:
        return "zai"
    if "aliyuncs" in url or "dashscope" in url:
        return "qwen"
    if "openrouter" in url:
        return "openrouter"
    if "api.x.ai" in url or "x.ai" in url:
        return "xai"
    if "deepseek" in url:
        return "deepseek"
    if "api.openai.com" in url:
        return "openai"
    return "generic"


def reasoning_payload(base_url: str, model_id: str, effort: str | None) -> dict[str, Any]:
    """Payload-фрагмент для выбранного вендора. Пустой dict — reasoning не слать.

    effort: 'low' | 'medium' | 'high' | 'none' | None.
    """
    effort = (effort or "").strip().lower()
    vendor = _vendor(base_url)
    model = (model_id or "").lower()

    if vendor == "zai":
        # GLM: thinking бинарный, градации усилия вендор не различает.
        if effort == "none":
            return {"thinking": {"type": "disabled"}}
        if effort in _EFFORTS:
            return {"thinking": {"type": "enabled"}}
        return {}

    if vendor == "qwen":
        if effort == "none":
            return {"enable_thinking": False}
        if effort in _EFFORTS:
            return {"enable_thinking": True}
        return {}

    if vendor == "openrouter":
        if effort == "none":
            return {"reasoning": {"enabled": False}}
        if effort in _EFFORTS:
            return {"reasoning": {"effort": effort}}
        return {}

    if vendor == "xai":
        # grok-4 думает всегда и параметр не принимает; grok-3-mini понимает low|high.
        if model.startswith("grok-3-mini") and effort in {"low", "high"}:
            return {"reasoning_effort": effort}
        if model.startswith("grok-3-mini") and effort == "medium":
            return {"reasoning_effort": "high"}
        return {}

    if vendor == "deepseek":
        # Режим выбирается моделью (deepseek-reasoner), параметров reasoning нет.
        return {}

    # OpenAI и прочие: reasoning_effort low|medium|high; 'none' — параметра нет.
    if effort in _EFFORTS:
        return {"reasoning_effort": effort}
    return {}
