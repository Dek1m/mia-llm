"""Tests for LLM Agents — сид Build/Plan, CRUD, ограничения."""
from __future__ import annotations

import pytest

from modules.llm.repository import LLMRepository
from modules.llm.provider import LLMProvider, ForbiddenError, NotFoundError
from modules.llm.config import LLMConfig, LLMProviderConfig
from modules.llm.providers.registry import ProviderRegistry
from modules.llm.tests.conftest import FakeLLMProvider


@pytest.fixture
def repo(mock_pool) -> LLMRepository:
    return LLMRepository(mock_pool)


@pytest.fixture
def provider(mock_pool):
    """LLMProvider с фейковым провайдером (без реального HTTP)."""
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
class TestSeedSystemAgents:
    async def test_seed_build_and_plan(self, repo: LLMRepository):
        """Системные агенты Build и Plan создаются идемпотентно."""
        await repo.seed_system_agents()
        build = await repo.get_agent_by_name("build")
        plan = await repo.get_agent_by_name("plan")
        assert build is not None
        assert build["agent_type"] == "system"
        assert build["system_prompt"] is not None
        assert "программировани" in build["system_prompt"]
        assert plan is not None
        assert plan["agent_type"] == "system"
        assert "планирован" in plan["system_prompt"]

    async def test_seed_is_idempotent(self, repo: LLMRepository):
        """Повторный сид не дублирует агентов."""
        await repo.seed_system_agents()
        await repo.seed_system_agents()
        count = await repo.count_agents_by_type("system")
        assert count == 2


@pytest.mark.asyncio
class TestAgentCRUD:
    async def test_create_agent(self, repo: LLMRepository):
        agent = await repo.create_agent(
            name="my-agent",
            agent_type="user",
            description="Test agent",
            system_prompt="You are helpful.",
        )
        assert agent["name"] == "my-agent"
        assert agent["agent_type"] == "user"

    async def test_get_agent(self, repo: LLMRepository):
        agent = await repo.create_agent(name="test", agent_type="user")
        found = await repo.get_agent(agent["id"])
        assert found is not None
        assert found["name"] == "test"

    async def test_get_agent_by_name(self, repo: LLMRepository):
        await repo.create_agent(name="named-agent", agent_type="user")
        found = await repo.get_agent_by_name("named-agent")
        assert found is not None

    async def test_update_agent(self, repo: LLMRepository):
        agent = await repo.create_agent(name="test", agent_type="user")
        updated = await repo.update_agent(agent["id"], {"description": "Updated"})
        assert updated is not None
        assert updated["description"] == "Updated"

    async def test_delete_agent(self, repo: LLMRepository):
        agent = await repo.create_agent(name="to-delete", agent_type="user")
        assert await repo.delete_agent(agent["id"]) is True
        assert await repo.get_agent(agent["id"]) is None

    async def test_list_agents(self, repo: LLMRepository):
        await repo.create_agent(name="a1", agent_type="user")
        await repo.create_agent(name="a2", agent_type="system")
        items, total = await repo.list_agents()
        assert total == 2

    async def test_list_agents_filter_type(self, repo: LLMRepository):
        await repo.create_agent(name="a1", agent_type="user")
        await repo.create_agent(name="a2", agent_type="system")
        items, total = await repo.list_agents(agent_type="user")
        assert total == 1


@pytest.mark.asyncio
class TestProviderAgentMethods:
    async def test_agents_list(self, provider: LLMProvider):
        result = await provider.agents()
        assert "items" in result
        assert "total" in result

    async def test_agent_not_found(self, provider: LLMProvider):
        with pytest.raises(NotFoundError):
            await provider.agent("nonexistent")

    async def test_create_agent(self, provider: LLMProvider):
        agent = await provider.create_agent(
            name="test-agent",
            agent_type="user",
            description="Test",
        )
        assert agent["name"] == "test-agent"

    async def test_create_system_agent_forbidden(self, provider: LLMProvider):
        with pytest.raises(ForbiddenError, match="system"):
            await provider.create_agent(
                name="bad",
                agent_type="system",
            )

    async def test_delete_system_agent_forbidden(self, provider: LLMProvider, repo: LLMRepository):
        # Создаём системного агента напрямую в БД (минуя provider)
        await repo.seed_system_agents()
        build = await repo.get_agent_by_name("build")
        assert build is not None
        # Не можем удалить через delete_agent
        with pytest.raises(ForbiddenError, match="system"):
            await provider.delete_agent(build["id"])

    async def test_update_system_agent_forbidden(self, provider: LLMProvider, repo: LLMRepository):
        await repo.seed_system_agents()
        build = await repo.get_agent_by_name("build")
        assert build is not None
        with pytest.raises(ForbiddenError, match="system"):
            await provider.update_agent(build["id"], {"description": "hacked"})

    async def test_delete_nonexistent_agent(self, provider: LLMProvider):
        with pytest.raises(NotFoundError):
            await provider.delete_agent("nonexistent")
