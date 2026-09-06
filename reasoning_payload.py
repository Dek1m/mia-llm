"""Вендор-адаптер режимов reasoning для OpenAI-совместимых чатов.

Единого стандарта нет: OpenAI/o-серии ждут reasoning_effort, GLM — thinking.type,
Qwen — enable_thinking, OpenRouter — reasoning.effort, DeepSeek выбирает режим
моделью. Здесь один вход (base_url, model_id, effort) — один выход (payload-dict).
"""
from __future__ import annotations

from typing import Any

_EFFORTS = {"min", "low", "medium", "high", "max"}


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
        # grok-4.x: шкала min|low|high|max; grok-3-mini: только low|high.
        if model.startswith("grok-4"):
            if effort == "medium":
                return {"reasoning_effort": "high"}
            if effort in _EFFORTS:
                return {"reasoning_effort": effort}
            return {}
        if model.startswith("grok-3-mini"):
            if effort in {"low", "high", "max"}:
                return {"reasoning_effort": "high" if effort == "max" else effort}
            if effort in {"min", "medium"}:
                return {"reasoning_effort": "low"}
            return {}
        # grok-4 всегда думает, параметр не принимает; прочие — молчим.
        return {}

    if vendor == "deepseek":
        # Режим выбирается моделью (deepseek-reasoner), параметров reasoning нет.
        return {}

    # OpenAI и прочие: reasoning_effort low|medium|high; 'none' — параметра нет.
    if effort in {"low", "medium", "high"}:
        return {"reasoning_effort": effort}
    return {}


# ── Probe шкалы через ошибку валидации ────────────────────────
# Вендор перечисляет допустимые уровни в тексте 400 на невалидное значение.

PROBE_EFFORT = "llm"

_PROBE_EFFORT_MAP = {
    "none": "none",
    "auto": "auto",
    "default": "default",
    "minimal": "min",
    "min": "min",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "max",
    "maximum": "max",
}
_PROBE_EFFORT_OUT = {"none", "auto", "default", "min", "low", "medium", "high", "max"}


def reasoning_probe_payload(base_url: str, model_id: str) -> dict[str, Any] | None:
    """Chat-запрос с заведомо невалидным reasoning-значением вендора.

    Ошибка валидации перечислит допустимые уровни — их вытащит parse_efforts_from_error.
    None — параметр бинарный (qwen) или отсутствует (deepseek): шкалы нет.
    """
    vendor = _vendor(base_url)
    base: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    if vendor == "zai":
        return {**base, "thinking": {"type": PROBE_EFFORT}}
    if vendor in {"qwen", "deepseek"}:
        return None
    if vendor == "openrouter":
        return {**base, "reasoning": {"effort": PROBE_EFFORT}}
    # xai / openai / generic
    return {**base, "reasoning_effort": PROBE_EFFORT}


def parse_efforts_from_error(message: str) -> list[str] | None:
    """Вытащить уровни reasoning из текста ошибки валидации. None — не удалось."""
    import re

    text = (message or "").lower()
    tokens: list[str] = re.findall(r"['\"`]([a-z_]+)['\"`]", text)
    if not tokens:
        m = re.search(
            r"(?:supported values|allowed values|must be one of|expected one of|one of)"
            r"[:\s]+([a-z0-9_,\s]+)",
            text,
        )
        if m:
            tokens = [t.strip() for t in m.group(1).split(",") if t.strip()]
    mapped: list[str] = []
    for token in tokens:
        norm = _PROBE_EFFORT_MAP.get(token)
        if norm and norm in _PROBE_EFFORT_OUT and norm not in mapped:
            mapped.append(norm)
    # Осмысленная шкала — минимум два уровня; иначе это шум из чужих кавычек.
    return mapped if len(mapped) >= 2 else None
