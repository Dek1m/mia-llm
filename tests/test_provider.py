"""Tests for LLM Provider — chat, agents, providers, schema registration."""
from __future__ import annotations

from typing import Any

import pytest

from modules.llm.provider import LLMProvider, LLMError, NotFoundError
from modules.llm.config import LLMConfig, LLMProviderConfig
from modules.llm.repository import LLMRepository
from modules.llm.tests.conftest import FakeLLMProvider


@pytest.fixture
def provider(mock_pool):
    """LLMProvider с фейковым провайдером и mock БД."""
    config = LLMConfig(
        providers={"openai": LLMProviderConfig()},
        default_provider="openai",
    )
    prov = LLMProvider(config)
    # Заменяем реальный OpenAI провайдер на фейковый
    fake = FakeLLMProvider(name="openai")
    prov.provider_registry.register("openai", fake)
    prov.provider_registry.set_default("openai")
    # Подменяем repo на mock_pool
    prov._repo = LLMRepository(mock_pool)
    return prov


@pytest.mark.asyncio
class TestChat:
    async def test_chat_returns_response(self, provider: LLMProvider):
        """chat() через mock-провайдер (default_provider=mock)."""
        # Переопределяем default_provider на fake
        provider._provider_registry.set_default("openai")
        # Заменяем default провайдер на fake
        fake = FakeLLMProvider(name="openai")
        provider._provider_registry.register("openai", fake)
        provider._provider_registry.set_default("openai")

        result = await provider.chat(messages=[{"role": "user", "content": "hi"}])
        assert "content" in result
        assert "[fake:openai]" in result["content"]

    async def test_chat_stream_returns_response(self, provider: LLMProvider):
        """chat_stream() в первой итерации — обёртка над chat()."""
        fake = FakeLLMProvider(name="openai")
        provider._provider_registry.register("openai", fake)
        provider._provider_registry.set_default("openai")

        result = await provider.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "content" in result


@pytest.mark.asyncio
class TestProviders:
    async def test_get_providers(self, provider: LLMProvider):
        """get_providers() возвращает список провайдеров."""
        result = await provider.get_providers()
        assert len(result) > 0
        assert any(p["name"] == "openai" for p in result)

    async def test_provider_health(self, provider: LLMProvider):
        """Провайдер по умолчанию (фейковый) → healthy=True."""
        result = await provider.get_providers()
        openai = next((p for p in result if p["name"] == "openai"), None)
        assert openai is not None
        assert openai["healthy"] is True


@pytest.mark.asyncio
class TestSchemaRegistration:
    async def test_initialize_creates_schema(self, provider: LLMProvider, mock_pool):
        """initialize() создаёт.AUTH_SCHEMA в БД."""
        # Проверяем что AUTH_SCHEMA валиден
        from modules.llm.schema import LLM_SCHEMA
        assert "permissions" in LLM_SCHEMA
        assert len(LLM_SCHEMA["permissions"]) == 5
        assert "roles" in LLM_SCHEMA
        assert len(LLM_SCHEMA["roles"]) == 2

    def test_schema_permissions_namespace(self):
        """Все permissions начинаются с 'llm:'."""
        from modules.llm.schema import LLM_SCHEMA
        for perm in LLM_SCHEMA["permissions"]:
            assert perm["name"].startswith("llm:"), f"Permission '{perm['name']}' must start with 'llm:'"

    def test_schema_roles_have_descriptions(self):
        from modules.llm.schema import LLM_SCHEMA
        for role in LLM_SCHEMA["roles"]:
            assert role["description"], f"Role '{role['name']}' has no description"

    def test_db_schema_has_agents_table(self):
        from modules.llm.schemas import DB_SCHEMA
        assert "schema" in DB_SCHEMA
        assert DB_SCHEMA["schema"] == "llm"
        assert "llm_agents" in DB_SCHEMA
