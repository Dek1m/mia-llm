"""LLM Module Configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = ["LLMConfig"]


@dataclass
class LLMConfig:
    """Конфигурация LLM-модуля."""

    default_provider: str = "openai_compatible"
    base_url: str = "http://localhost:8080/v1"
    api_key: str = ""
    default_model: str = "qwen3"
    default_temperature: float = 0.7
    default_max_tokens: int = 4096
    default_top_p: float = 1.0
    timeout: float = 120.0
    max_retries: int = 2
    stream_by_default: bool = False

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            default_provider=os.getenv("LLM_DEFAULT_PROVIDER", "openai_compatible"),
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"),
            api_key=os.getenv("LLM_API_KEY", ""),
            default_model=os.getenv("LLM_DEFAULT_MODEL", "qwen3"),
            default_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.7")),
            default_max_tokens=int(os.getenv("LLM_DEFAULT_MAX_TOKENS", "4096")),
            default_top_p=float(os.getenv("LLM_DEFAULT_TOP_P", "1.0")),
            timeout=float(os.getenv("LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            stream_by_default=os.getenv("LLM_STREAM_BY_DEFAULT", "false").lower() == "true",
        )
