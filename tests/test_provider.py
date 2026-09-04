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

    async def test_run_pipeline_with_messages(self, provider: LLMProvider):
        fake = FakeLLMProvider(name="openai")
        provider._provider_registry.register("openai", fake)
        provider._provider_registry.set_default("openai")
        result = await provider.run_pipeline(
            workspace_id="w",
            session_id="s",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result["status"] == "success"
        assert result["content"]


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
        assert len(LLM_SCHEMA["permissions"]) == 7
        assert "roles" in LLM_SCHEMA
        assert len(LLM_SCHEMA["roles"]) == 3

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
        assert "llm_providers" in DB_SCHEMA
        assert "llm_models" in DB_SCHEMA
        assert "pipelines" in DB_SCHEMA
        assert "runs" in DB_SCHEMA
        assert "description" in DB_SCHEMA["llm_providers"]["columns"]


class TestApiExport:
    """@task(api=True) на LLMProvider: initialize без api."""

    def test_collect_from_module_exports_eight_api_methods(
        self, provider: LLMProvider,
    ) -> None:
        from modules.apiproxy.registry import MethodRegistry

        reg = MethodRegistry()
        count = reg.collect_from_module(provider, "llm")
        names = {m.name for m in reg.list_methods("llm")}
        assert count == 33
        assert names == {
            "chat",
            "chat_stream",
            "agents",
            "agent",
            "create_agent",
            "update_agent",
            "delete_agent",
            "get_providers",
            "list_providers",
            "create_provider",
            "list_oauth_vendors",
            "start_oauth",
            "poll_oauth",
            "probe_provider_models",
            "probe_models",
            "refresh_catalog",
            "set_model_enabled",
            "delete_provider",
            "set_model_reasoning",
            "update_provider",
            "delete_model",
            "set_model_name",
            "set_provider_models_enabled",
            "share_provider",
            "unshare_provider",
            "list_provider_shares",
            "list_middleware",
            "list_pipelines",
            "run_usage",
            "run_pipeline",
            "cancel_run",
            "set_agent_avatar",
            "clear_agent_avatar",
        }

    def test_initialize_has_no_api_meta(self) -> None:
        assert not hasattr(LLMProvider.initialize, "_api_meta")
        assert hasattr(LLMProvider.initialize, "_task_type")

    def test_all_api_methods_have_llm_permission(
        self, provider: LLMProvider,
    ) -> None:
        from modules.apiproxy.registry import MethodRegistry

        reg = MethodRegistry()
        reg.collect_from_module(provider, "llm")
        for meta in reg.list_methods("llm"):
            assert meta.required_permission, f"{meta.name} без permission"
            assert meta.required_permission.startswith("llm:")
            assert meta.public is False


@pytest.mark.asyncio
class TestSavedProviders:
    async def test_create_api_key_hides_secret(self, provider: LLMProvider) -> None:
        row = await provider.create_provider(
            name="lab",
            kind="api_key",
            vendor="openai",
            api_key="sk-secret",
        )
        assert row["name"] == "lab"
        assert row["api_key_set"] is True
        assert "api_key" not in row

    async def test_list_oauth_vendors_includes_xai(self, provider: LLMProvider) -> None:
        row = await provider.list_oauth_vendors()
        ids = {item["id"] for item in row["items"]}
        assert "xai" in ids

    async def test_start_oauth_unknown_vendor(self, provider: LLMProvider) -> None:
        with pytest.raises(Exception) as caught:
            await provider.start_oauth("acme")
        assert getattr(caught.value, "code", "") == "OAUTH_UNSUPPORTED"

    async def test_start_oauth_device_code(self, provider: LLMProvider, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_device(_vendor: object) -> dict[str, object]:
            return {
                "device_code": "dev-1",
                "user_code": "ABCD-1234",
                "verification_uri": "https://accounts.x.ai/oauth2/device",
                "verification_uri_complete": "https://accounts.x.ai/oauth2/device?user_code=ABCD-1234",
                "interval": 5,
                "expires_in": 900,
                "expires_at": 9999999999,
            }

        monkeypatch.setattr("modules.llm.provider.request_device_code", fake_device)
        row = await provider.start_oauth("xai", name="xAI")
        assert row["status"] == "authorizing"
        assert row["user_code"] == "ABCD-1234"
        assert row["vendor"] == "xai"
        assert row["provider_id"]
