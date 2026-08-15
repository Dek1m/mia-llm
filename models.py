"""Модели данных LLM-модуля."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class AgentDefinition:
    id: str
    name: str
    system_prompt: str
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ChatMessage:
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


@dataclass
class ChatResponse:
    id: str
    content: str | None
    role: str = "assistant"
    model: str | None = None
    finish_reason: str | None = None
    tool_calls: list[dict] | None = None
    usage: dict[str, int] | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def from_openai(cls, data: dict[str, Any]) -> ChatResponse:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return cls(
            id=data.get("id") or str(uuid4()),
            content=message.get("content"),
            role=message.get("role", "assistant"),
            model=data.get("model"),
            finish_reason=choice.get("finish_reason"),
            tool_calls=message.get("tool_calls"),
            usage=data.get("usage"),
            raw=data,
        )


@dataclass
class StreamChunk:
    delta: str
    finish_reason: str | None = None
    tool_calls: list[dict] | None = None
    raw: dict[str, Any] | None = None
