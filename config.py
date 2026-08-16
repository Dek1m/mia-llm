"""LLM Module Configuration."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

__all__ = ["LLMConfig"]


@dataclass
class LLMProviderConfig:
    """Конфигурация одного LLM-провайдера."""
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    default_model: str = "gpt-4o-mini"
    timeout: float = 120.0


@dataclass
class LLMConfig:
    """Конфигурация LLM-модуля."""

    # Провайдеры: name → config
    providers: dict[str, LLMProviderConfig] = field(default_factory=dict)

    # Default и fallback
    default_provider: str = "openai"
    fallback_provider: str = ""

    # Глобальные настройки
    default_temperature: float = 0.7
    default_max_tokens: int = 4096
    default_top_p: float = 1.0

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Создать конфигурацию из переменных окружения."""
        # Провайдеры из JSON
        providers_raw = os.getenv("MIA_LLM_PROVIDERS", "")
        providers: dict[str, LLMProviderConfig] = {}
        if providers_raw:
            try:
                providers_data = json.loads(providers_raw)
                for name, cfg in providers_data.items():
                    providers[name] = LLMProviderConfig(
                        base_url=cfg.get("base_url", "https://api.openai.com/v1"),
                        api_key=cfg.get("api_key", ""),
                        default_model=cfg.get("default_model", "gpt-4o-mini"),
                        timeout=float(cfg.get("timeout", "120")),
                    )
            except (json.JSONDecodeError, KeyError):
                pass

        # Если провайдеры не заданы — создаём дефолтного из env
        if not providers:
            providers["openai"] = LLMProviderConfig(
                base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.getenv("LLM_API_KEY", ""),
                default_model=os.getenv("LLM_DEFAULT_MODEL", "gpt-4o-mini"),
                timeout=float(os.getenv("LLM_TIMEOUT", "120")),
            )

        return cls(
            providers=providers,
            default_provider=os.getenv("MIA_LLM_DEFAULT_PROVIDER", "openai"),
            fallback_provider=os.getenv("MIA_LLM_FALLBACK_PROVIDER", ""),
            default_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.7")),
            default_max_tokens=int(os.getenv("LLM_DEFAULT_MAX_TOKENS", "4096")),
            default_top_p=float(os.getenv("LLM_DEFAULT_TOP_P", "1.0")),
        )
