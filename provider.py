"""LLM Provider — основной провайдер модуля LLM.

Интеграция с БД (агенты), провайдерами (chat), auth (permissions).
"""
from __future__ import annotations

import time
from typing import Any

from core.task_decorator import task

from .config import LLMConfig
from .repository import LLMRepository
from .schema import LLM_SCHEMA
from .schemas import DB_SCHEMA
from .providers.openai import OpenAIProvider
from .providers.registry import ProviderRegistry
from .oauth import (
    bearer_from_stored,
    get_vendor,
    list_vendors,
    pack_device,
    pack_tokens,
    poll_device_token,
    request_device_code,
    unpack_secret,
)
from .secrets import encrypt_secret
from modules.auth.validators import ForbiddenAvatarError, decode_avatar


__all__ = ["LLMProvider"]


class LLMError(Exception):
    """Базовая ошибка LLM-модуля. message — в лог, human — клиенту."""

    def __init__(self, message: str, code: str = "LLM_ERROR", *, human: str | None = None) -> None:
        self.code = code
        self.human = human or message
        super().__init__(message)


def _is_duplicate_name(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).lower()
        if "llm_providers_name_key" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def _raise_duplicate_name(name: str, exc: BaseException) -> None:
    raise LLMError(
        f"duplicate provider name {name!r}: {exc}",
        "DUPLICATE_NAME",
        human="A provider with this name already exists",
    ) from exc


class NotFoundError(LLMError):
    def __init__(self, entity: str = "Resource") -> None:
        super().__init__(f"{entity} not found", "NOT_FOUND")


class ForbiddenError(LLMError):
    def __init__(self, message: str = "Forbidden") -> None:
        super().__init__(message, "FORBIDDEN")


class LLMProvider:
    """Провайдер LLM.

    Предоставляет:
    - Вызов LLM через провайдеры с fallback
    - Управление агентами (CRUD в БД)
    - Просмотр провайдеров и моделей
    """

    _AGENT_KINDS = frozenset({"agent", "subagent", "cronagent", "user"})

    def __init__(self, config: LLMConfig, log: Any = None) -> None:
        self._config = config
        self._log = log
        self._repo: LLMRepository | None = None
        self._system_repo: LLMRepository | None = None
        self._database: Any = None
        self._state: Any = None
        self._provider_registry = ProviderRegistry(log=log)
        self._init_providers()

    def _init_providers(self) -> None:
        """Создать и зарегистрировать провайдеры из конфига."""
        for name, pcfg in self._config.providers.items():
            provider = OpenAIProvider(
                name=name,
                base_url=pcfg.base_url,
                api_key=pcfg.api_key,
                default_model=pcfg.default_model,
                timeout=pcfg.timeout,
                log=self._log,
            )
            self._provider_registry.register(name, provider)

        if self._config.default_provider:
            self._provider_registry.set_default(self._config.default_provider)
        if self._config.fallback_provider:
            self._provider_registry.set_fallback(self._config.fallback_provider)

    @property
    def repository(self) -> LLMRepository | None:
        return self._repo

    @property
    def provider_registry(self) -> ProviderRegistry:
        return self._provider_registry

    def bind_pool(self, pool: Any) -> None:
        """Пул на воркере без повторного apply_schema."""
        self._repo = LLMRepository(pool, log=self._log)
        self._system_repo = self._repo

    def bind_runtime(self, state: Any, database: Any) -> None:
        self._state = state
        self._database = database
        self.bind_pool(database.pool)

    def _open_user_repo(self, user_id: str) -> LLMRepository:
        from copy import deepcopy

        from modules.workspace.schemas import user_dbname

        if self._state is not None:
            self._state.workspace(user=user_id)
        dbname = user_dbname(user_id)
        pool = self._database.get_pool(dbname)
        self._database.register_schema(
            "llm",
            deepcopy(DB_SCHEMA),
            schema_name="llm",
            pool=pool,
        )
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_models_provider_model "
                    "ON llm.llm_models (provider_id, model_id)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_llm_models_provider ON llm.llm_models (provider_id)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_llm_providers_kind ON llm.llm_providers (kind)"
                )
        return LLMRepository(pool, log=self._log)

    def _providers_repo(self, user_id: str | None, *, common: bool = False) -> LLMRepository:
        if common:
            if self._system_repo is None:
                raise LLMError("LLM not initialized (no DB pool)")
            return self._system_repo
        if user_id and self._database is not None and self._state is not None:
            return self._open_user_repo(str(user_id))
        if self._repo is None:
            raise LLMError("LLM not initialized (no DB pool)")
        return self._repo

    async def _can_manage_common(self, user_id: str | None) -> bool:
        if not user_id or self._state is None:
            return False
        from modules.auth.provider import AuthProvider

        auth = self._state.services.resolve(AuthProvider)
        return bool(await auth.check_permission(str(user_id), "llm:provider_share"))

    async def _locate_provider(
        self, user_id: str | None, provider_id: str,
    ) -> tuple[LLMRepository, dict[str, Any], bool]:
        if user_id and self._database is not None and self._state is not None:
            urepo = self._open_user_repo(str(user_id))
            row = await urepo.get_provider(provider_id)
            if row:
                return urepo, row, False
        if self._system_repo is not None:
            row = await self._system_repo.get_provider(provider_id)
            if row:
                return self._system_repo, row, True
        raise NotFoundError("Provider")

    @task(type="database")
    async def initialize(self, state: Any) -> None:
        """Регистрация БД-схемы и AUTH_SCHEMA."""
        self.initialize_sync(state)

    def initialize_sync(self, state: Any) -> None:
        """Синхронная версия initialize для on_load. Не через @task."""
        from copy import deepcopy

        from modules.auth.provider import AuthProvider
        from modules.db.provider import DatabaseProvider

        db_provider = state.services.resolve(DatabaseProvider)
        self._repo = LLMRepository(db_provider.pool, log=self._log)
        db_provider.register_schema(
            "llm",
            deepcopy(DB_SCHEMA),
            schema_name="llm",
            ddl_dir=str(__import__("pathlib").Path(__file__).resolve().parent / "ddl"),
        )
        try:
            auth = state.services.resolve(AuthProvider)
            if auth.registry is not None:
                auth.registry.register_sync("llm", LLM_SCHEMA, is_builtin=False)
            db_provider.execute(
                "INSERT INTO auth.group_roles (group_id, role_id) "
                "SELECT g.id, r.id FROM auth.groups g CROSS JOIN auth.roles r "
                "WHERE g.name = %s AND r.name = %s "
                "ON CONFLICT (group_id, role_id) DO NOTHING",
                "Everyone",
                "llm_user",
            )
        except Exception as exc:
            if self._log is not None:
                self._log.warning("llm_auth_schema_skipped", extra={"error": str(exc)})
        if self._repo is not None:
            try:
                db_provider.execute(
                    "ALTER TABLE llm.llm_providers ADD COLUMN IF NOT EXISTS description TEXT",
                )
                db_provider.execute(
                    "ALTER TABLE llm.llm_models "
                    "ADD COLUMN IF NOT EXISTS supports_reasoning BOOLEAN NOT NULL DEFAULT FALSE",
                )
                db_provider.execute(
                    "ALTER TABLE llm.llm_models "
                    "ADD COLUMN IF NOT EXISTS reasoning_enabled BOOLEAN NOT NULL DEFAULT FALSE",
                )
                db_provider.execute(
                    "ALTER TABLE llm.llm_models ADD COLUMN IF NOT EXISTS reasoning_effort TEXT",
                )
                self._repo.reencrypt_api_keys_sync()
            except Exception as exc:
                if self._log is not None:
                    self._log.warning("llm_providers_alter_failed", extra={"error": str(exc)})
        self._repo.seed_system_agents_sync()
        if self._log is not None:
            self._log.info("LLM schema registered, system agents seeded")

    # ── Chat ────────────────────────────────────────────

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Общий путь chat / chat_stream. Не через @task — без nested dispatch."""
        return await self._provider_registry.chat_with_fallback(
            messages=messages,
            model=model,
            **kwargs,
        )

    @task(
        type="network",
        api=True,
        permission="llm:chat",
        name="chat",
        description="Вызов LLM через chat completions",
        args={"messages": "list", "model": "str", "provider": "str"},
        return_type="dict",
    )
    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Вызвать LLM через провайдер с fallback."""
        return await self._chat(messages, model=model, **kwargs)

    @task(
        type="network",
        api=True,
        permission="llm:chat_stream",
        name="chat_stream",
        description="Потоковый вывод LLM (пока синхронный обёртка)",
        args={"messages": "list", "model": "str"},
        return_type="dict",
    )
    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Пока без SSE — тот же путь, что chat."""
        return await self._chat(messages, model=model, **kwargs)

    # ── Agents ──────────────────────────────────────────

    @task(
        type="database",
        api=True,
        permission="llm:agent_list",
        name="agents",
        description="Список всех агентов (системные + пользовательские + workspace)",
        args={"agent_type": "str", "workspace_id": "str", "offset": "int", "limit": "int"},
        return_type="dict",
    )
    async def agents(
        self,
        agent_type: str | None = None,
        workspace_id: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Получить список агентов."""
        if self._repo is None:
            raise LLMError("LLM not initialized (no DB pool)")
        items, total = await self._repo.list_agents(
            agent_type=agent_type, workspace_id=workspace_id,
            offset=offset, limit=limit,
        )
        return {"items": items, "total": total, "offset": offset, "limit": limit}

    @task(
        type="database",
        api=True,
        permission="llm:agent_list",
        name="agent",
        description="Получить информацию об агенте",
        args={"agent_id": "str"},
        return_type="dict",
    )
    async def agent(self, agent_id: str) -> dict[str, Any]:
        """Получить агента по ID."""
        if self._repo is None:
            raise LLMError("LLM not initialized (no DB pool)")
        row = await self._repo.get_agent(agent_id)
        if not row:
            raise NotFoundError("Agent")
        return row

    @task(
        type="database",
        api=True,
        permission="llm:agent_manage",
        name="create_agent",
        description="Создать нового агента (пользовательский/workspace)",
        args={
            "name": "str", "agent_type": "str", "description": "str",
            "system_prompt": "str", "model": "str", "workspace_id": "str",
        },
        return_type="dict",
    )
    async def create_agent(
        self,
        name: str,
        agent_type: str = "agent",
        description: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        workspace_id: str | None = None,
        owner_id: str | None = None,
        _session_user_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Создать агента."""
        if self._repo is None:
            raise LLMError("LLM not initialized (no DB pool)")

        cleaned = (name or "").strip()
        if not cleaned:
            raise LLMError("Name is required", "VALIDATION", human="Name is required")
        if agent_type == "system" or agent_type not in self._AGENT_KINDS:
            raise ForbiddenError("Cannot create system agents manually")
        existing = await self._repo.get_agent_by_name(cleaned)
        if existing:
            raise LLMError("Agent already exists", "AGENT_EXISTS", human="Name already exists")

        row = await self._repo.create_agent(
            name=cleaned,
            agent_type=agent_type,
            description=description,
            system_prompt=system_prompt,
            model=model,
            workspace_id=workspace_id,
            owner_id=owner_id or _session_user_id,
        )
        return row

    @task(
        type="database",
        api=True,
        permission="llm:agent_manage",
        name="update_agent",
        description="Обновить агента",
        args={"agent_id": "str", "data": "dict"},
        return_type="dict",
    )
    async def update_agent(
        self,
        agent_id: str,
        data: dict[str, Any] | None = None,
        name: str | None = None,
        agent_type: str | None = None,
        description: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Обновить агента."""
        if self._repo is None:
            raise LLMError("LLM not initialized (no DB pool)")

        row = await self._repo.get_agent(agent_id)
        if not row:
            raise NotFoundError("Agent")
        if row.get("agent_type") == "system":
            raise ForbiddenError("Cannot modify system agents")

        patch = dict(data or {})
        for key, value in (
            ("name", name),
            ("agent_type", agent_type),
            ("description", description),
            ("system_prompt", system_prompt),
            ("model", model),
        ):
            if value is not None:
                patch[key] = value
        if patch.get("agent_type") == "system":
            raise ForbiddenError("Cannot modify system agents")
        if patch.get("agent_type") and patch["agent_type"] not in self._AGENT_KINDS:
            raise ForbiddenError("Unknown agent type")
        if "name" in patch:
            cleaned = str(patch["name"]).strip()
            if not cleaned:
                raise LLMError("Name is required", "VALIDATION", human="Name is required")
            patch["name"] = cleaned

        result = await self._repo.update_agent(agent_id, patch)
        if not result:
            raise NotFoundError("Agent")
        return result

    @task(
        type="database",
        api=True,
        permission="llm:agent_manage",
        name="delete_agent",
        description="Удалить агента",
        args={"agent_id": "str"},
        return_type="bool",
    )
    async def delete_agent(self, agent_id: str) -> bool:
        """Удалить агента."""
        if self._repo is None:
            raise LLMError("LLM not initialized (no DB pool)")

        row = await self._repo.get_agent(agent_id)
        if not row:
            raise NotFoundError("Agent")
        if row.get("agent_type") == "system":
            raise ForbiddenError("Cannot delete system agents")

        return await self._repo.delete_agent(agent_id)

    @task(
        type="cpu",
        api=True,
        permission="llm:agent_manage",
        name="set_agent_avatar",
        description="Загрузить аватар агента (jpeg/png/webp, не SVG, ≤256 KiB)",
        args={"agent_id": "str", "image_b64": "str", "content_type": "str"},
        return_type="dict",
    )
    async def set_agent_avatar(
        self, agent_id: str, image_b64: str, content_type: str,
    ) -> dict[str, Any]:
        if self._repo is None:
            raise LLMError("LLM not initialized (no DB pool)")
        row = await self._repo.get_agent(agent_id)
        if not row:
            raise NotFoundError("Agent")
        try:
            raw = decode_avatar(image_b64, content_type)
        except ForbiddenAvatarError as exc:
            raise ForbiddenError(str(exc)) from exc
        mime = content_type.split(";")[0].strip().lower()
        await self._repo.upsert_agent_avatar(agent_id, raw, mime)
        return {"avatar_url": f"/api/v1/llm/agent_avatar?agent_id={agent_id}"}

    @task(
        type="database",
        api=True,
        permission="llm:agent_manage",
        name="clear_agent_avatar",
        args={"agent_id": "str"},
        return_type="dict",
    )
    async def clear_agent_avatar(self, agent_id: str) -> dict[str, Any]:
        if self._repo is None:
            raise LLMError("LLM not initialized (no DB pool)")
        await self._repo.delete_agent_avatar(agent_id)
        return {"ok": True}

    async def get_agent_avatar_bytes(self, agent_id: str) -> tuple[bytes, str] | None:
        if self._repo is None:
            return None
        row = await self._repo.get_agent_avatar(agent_id)
        if not row or row.get("bytes") is None:
            return None
        return bytes(row["bytes"]), str(row.get("content_type") or "application/octet-stream")

    # ── Providers ───────────────────────────────────────

    @task(
        type="cpu",
        api=True,
        permission="llm:config",
        name="get_providers",
        description="Список зарегистрированных LLM-провайдеров и их статус",
        args={},
        return_type="list",
    )
    async def get_providers(self) -> list[dict[str, Any]]:
        """Получить список провайдеров с health-check."""
        providers = self._provider_registry.list_providers()
        for p in providers:
            prov = self._provider_registry.get(p["name"])
            if prov:
                try:
                    p["healthy"] = await prov.health()
                except Exception:
                    p["healthy"] = False
            else:
                p["healthy"] = False
        return providers

    @task(
        type="database",
        api=True,
        permission="llm:config",
        name="list_providers",
        description="Свои провайдеры и расшаренные на группы",
        args={},
        return_type="dict",
    )
    async def list_providers(self, _session_user_id: str | None = None) -> dict[str, Any]:
        repo = self._providers_repo(_session_user_id)
        items = await repo.list_providers()
        for item in items:
            item["owned"] = True
            item["shared"] = False
            item["common"] = False
        seen = {str(row["id"]) for row in items}
        if _session_user_id and self._system_repo is not None and self._state is not None:
            if await self._can_manage_common(_session_user_id):
                for item in await self._system_repo.list_providers():
                    pid = str(item["id"])
                    if pid in seen:
                        continue
                    item["owned"] = True
                    item["shared"] = False
                    item["common"] = True
                    seen.add(pid)
                    items.append(item)
            else:
                items.extend(await self._shared_providers(str(_session_user_id), seen))
        return {"items": items}

    @task(
        type="database",
        api=True,
        permission="llm:provider_manage",
        name="create_provider",
        description="Создать провайдера: api_key или oauth",
        args={
            "name": "str",
            "kind": "str",
            "vendor": "str",
            "description": "str",
            "base_url": "str",
            "default_model": "str",
            "api_key": "str",
            "models": "list",
            "common": "bool",
        },
        return_type="dict",
    )
    async def create_provider(
        self,
        name: str,
        kind: str,
        vendor: str,
        description: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        api_key: str | None = None,
        models: list[dict[str, Any]] | None = None,
        common: bool = False,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        if common and not await self._can_manage_common(_session_user_id):
            raise ForbiddenError("Only llm_admin can add organization providers")
        repo = self._providers_repo(_session_user_id, common=bool(common))
        kind_norm = kind.strip().lower()
        vendor_norm = vendor.strip().lower()
        if kind_norm not in {"api_key", "oauth"}:
            raise LLMError("kind must be api_key or oauth", "INVALID_NAME")
        if kind_norm == "api_key" and base_url:
            _validate_base_url(base_url)
        try:
            row = await repo.create_provider(
                name=name.strip(),
                kind=kind_norm,
                vendor=vendor_norm,
                description=(description or "").strip() or None,
                base_url=base_url.strip() if isinstance(base_url, str) else base_url,
                default_model=default_model,
                api_key=_cipher_key(api_key),
                oauth_status="pending" if kind_norm == "oauth" else None,
                models=models,
            )
        except Exception as exc:
            if _is_duplicate_name(exc):
                _raise_duplicate_name(name.strip(), exc)
            raise
        row["owned"] = True
        row["shared"] = False
        row["common"] = bool(common)
        if self._log is not None:
            self._log.info(
                "llm_provider_created",
                extra={"name": name, "kind": kind_norm, "vendor": vendor_norm, "common": bool(common)},
            )
        return row

    @task(
        type="database",
        api=True,
        permission="llm:provider_manage",
        name="list_oauth_vendors",
        description="Список OAuth-вендоров с device-code",
        args={},
        return_type="dict",
    )
    async def list_oauth_vendors(self, _session_user_id: str | None = None) -> dict[str, Any]:
        return {"items": list_vendors()}

    @task(
        type="network",
        api=True,
        permission="llm:provider_manage",
        name="start_oauth",
        description="Начать device-code OAuth и создать pending-провайдера",
        args={"vendor": "str", "name": "str", "description": "str", "common": "bool"},
        return_type="dict",
    )
    async def start_oauth(
        self,
        vendor: str,
        name: str | None = None,
        description: str | None = None,
        common: bool = False,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        if common and not await self._can_manage_common(_session_user_id):
            raise ForbiddenError("Only llm_admin can add organization providers")
        repo = self._providers_repo(_session_user_id, common=bool(common))
        spec = get_vendor(vendor)
        if spec is None:
            raise LLMError(
                f"unsupported oauth vendor {vendor!r}",
                "OAUTH_UNSUPPORTED",
                human="This OAuth provider is not supported yet",
            )
        try:
            device = await request_device_code(spec)
        except Exception as exc:
            raise LLMError(str(exc), "OAUTH_DENIED", human="Could not start sign-in") from exc
        label = (name or "").strip() or spec.name
        blob = pack_device(device["device_code"], int(device["interval"]), int(device["expires_at"]))
        try:
            row = await repo.create_provider(
                name=label,
                kind="oauth",
                vendor=spec.id,
                description=(description or "").strip() or None,
                base_url=spec.base_url,
                api_key=blob,
                oauth_status="authorizing",
            )
        except Exception as exc:
            if _is_duplicate_name(exc):
                _raise_duplicate_name(label, exc)
            raise
        return {
            "provider_id": str(row["id"]),
            "vendor": spec.id,
            "status": "authorizing",
            "mode": "device_code",
            "user_code": device["user_code"],
            "verification_uri": device["verification_uri"],
            "verification_uri_complete": device["verification_uri_complete"],
            "interval": device["interval"],
            "expires_in": device["expires_in"],
            "provider": row,
        }

    @task(
        type="network",
        api=True,
        permission="llm:provider_manage",
        name="poll_oauth",
        description="Один шаг device-code poll",
        args={"provider_id": "str"},
        return_type="dict",
    )
    async def poll_oauth(
        self,
        provider_id: str,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        repo, row, _common = await self._locate_provider(_session_user_id, provider_id)
        spec = get_vendor(str(row.get("vendor") or ""))
        if spec is None:
            raise LLMError("unsupported oauth vendor", "OAUTH_UNSUPPORTED", human="This OAuth provider is not supported yet")
        secret = unpack_secret(row.get("api_key"))
        if secret and secret.get("kind") == "token":
            public = dict(row)
            public.pop("api_key", None)
            public["api_key_set"] = True
            public["models"] = await repo.list_models(provider_id)
            return {"status": "connected", "provider": public}
        if not secret or secret.get("kind") != "device":
            raise LLMError("oauth session missing", "OAUTH_EXPIRED", human="Sign-in expired. Start again")
        if int(secret.get("expires_at") or 0) and int(secret["expires_at"]) < int(time.time()):
            await repo.set_oauth_state(provider_id, row.get("api_key") or "", "expired")
            raise LLMError("device code expired", "OAUTH_EXPIRED", human="Sign-in expired. Start again")
        result = await poll_device_token(spec, str(secret.get("device_code") or ""))
        status = result.get("status")
        if status == "pending" or status == "slow_down":
            return {"status": "pending", "interval": int(secret.get("interval") or 5)}
        if status == "expired":
            await repo.set_oauth_state(provider_id, row.get("api_key") or "", "expired")
            raise LLMError("device code expired", "OAUTH_EXPIRED", human="Sign-in expired. Start again")
        if status == "denied":
            await repo.set_oauth_state(provider_id, row.get("api_key") or "", "denied")
            raise LLMError("access denied", "OAUTH_DENIED", human="Sign-in was denied")
        if status != "connected":
            raise LLMError(str(result.get("detail") or "oauth failed"), "OAUTH_DENIED", human="Could not finish sign-in")
        blob = pack_tokens(str(result["access"]), result.get("refresh"), int(result["expires_at"]))
        public = await repo.set_oauth_state(provider_id, blob, "connected")
        return {"status": "connected", "provider": public}

    @task(
        type="network",
        api=True,
        permission="llm:provider_manage",
        name="probe_provider_models",
        description="Каталог моделей сохранённого провайдера",
        args={"provider_id": "str"},
        return_type="dict",
    )
    async def probe_provider_models(
        self,
        provider_id: str,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        repo, row, _common = await self._locate_provider(_session_user_id, provider_id)
        token = bearer_from_stored(row.get("api_key"))
        base = row.get("base_url")
        if not token or not base:
            raise LLMError("sign-in is not finished", "OAUTH_PENDING", human="Finish sign-in first")
        items = await self._fetch_remote_models(str(base), token)
        return {"items": items}

    @task(
        type="database",
        api=True,
        permission="llm:provider_manage",
        name="delete_provider",
        description="Удалить провайдера, ключ и модели",
        args={"provider_id": "str"},
        return_type="dict",
    )
    async def delete_provider(
        self,
        provider_id: str,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        repo, _row, common = await self._locate_provider(_session_user_id, provider_id)
        if common and not await self._can_manage_common(_session_user_id):
            raise ForbiddenError("Cannot change an organization provider")
        deleted = await repo.delete_provider(provider_id)
        if not deleted:
            raise NotFoundError("Provider")
        return {"ok": True, "id": provider_id}

    @task(
        type="database",
        api=True,
        permission="llm:provider_manage",
        name="update_provider",
        description="Переименовать провайдера и обновить поля",
        args={
            "provider_id": "str",
            "name": "str",
            "description": "str",
            "base_url": "str",
            "api_key": "str",
            "models": "list",
        },
        return_type="dict",
    )
    async def update_provider(
        self,
        provider_id: str,
        name: str,
        description: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        models: list[dict[str, Any]] | None = None,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        repo, existing, common = await self._locate_provider(_session_user_id, provider_id)
        if common and not await self._can_manage_common(_session_user_id):
            raise ForbiddenError("Cannot change an organization provider")
        key = api_key.strip() if isinstance(api_key, str) else ""
        oauth = str(existing.get("kind") or "") == "oauth"
        if not key and not oauth:
            raise LLMError("API key is required", "INVALID_NAME")
        next_url = base_url.strip() if isinstance(base_url, str) and base_url.strip() else existing.get("base_url")
        if next_url:
            _validate_base_url(str(next_url))
        try:
            row = await repo.update_provider(
                provider_id=provider_id,
                name=name.strip(),
                description=(description or "").strip() or None,
                base_url=str(next_url) if next_url else None,
                api_key=_cipher_key(key) if key else None,
                models=models,
            )
        except Exception as exc:
            if _is_duplicate_name(exc):
                _raise_duplicate_name(name.strip(), exc)
            raise
        if not row:
            raise NotFoundError("Provider")
        return row

    @task(
        type="database",
        api=True,
        permission="llm:provider_manage",
        name="delete_model",
        description="Удалить модель провайдера",
        args={"model_id": "str"},
        return_type="dict",
    )
    async def delete_model(
        self,
        model_id: str,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        repo = self._providers_repo(_session_user_id)
        deleted = await repo.delete_model(model_id)
        if not deleted:
            raise NotFoundError("Model")
        return {"ok": True, "id": model_id}

    @task(
        type="network",
        api=True,
        permission="llm:provider_manage",
        name="probe_models",
        description="GET {base}/models по OpenAI-совместимому URL",
        args={"base_url": "str", "api_key": "str"},
        return_type="dict",
    )
    async def probe_models(
        self,
        base_url: str,
        api_key: str,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        _validate_base_url(base_url)
        items = await self._fetch_remote_models(base_url, api_key)
        return {"items": items}

    @task(
        type="network",
        api=True,
        permission="llm:config",
        name="refresh_catalog",
        description="Сверить сохранённые модели с вендором; пропавшие — в vanished",
        args={},
        return_type="dict",
    )
    async def refresh_catalog(self, _session_user_id: str | None = None) -> dict[str, Any]:
        repos = [self._providers_repo(_session_user_id)]
        if await self._can_manage_common(_session_user_id) and self._system_repo is not None:
            repos.append(self._system_repo)
        vanished: list[dict[str, Any]] = []
        for repo in repos:
            for row in await repo.list_providers_with_secrets():
                key = bearer_from_stored(row.get("api_key"))
                base = row.get("base_url")
                if not key or not base:
                    continue
                try:
                    remote_items = await self._fetch_remote_models(str(base), str(key))
                    remote = [str(item["id"]) for item in remote_items]
                except Exception as exc:
                    if self._log is not None:
                        self._log.warning(
                            "llm_catalog_refresh_failed",
                            extra={"provider_id": str(row.get("id")), "error": str(exc)},
                        )
                    continue
                missing = await repo.replace_remote_models(str(row["id"]), remote)
                for item in missing:
                    vanished.append(
                        {
                            "provider_id": str(row["id"]),
                            "provider_name": row.get("name"),
                            "model_id": item.get("model_id"),
                            "display_name": item.get("display_name"),
                        }
                    )
        return {"vanished": vanished}

    @task(
        type="database",
        api=True,
        permission="llm:provider_manage",
        name="set_model_enabled",
        description="Включить или выключить модель провайдера",
        args={"model_id": "str", "enabled": "bool"},
        return_type="dict",
    )
    async def set_model_enabled(
        self,
        model_id: str,
        enabled: bool,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        repo = self._providers_repo(_session_user_id)
        row = await repo.set_model_enabled(model_id, bool(enabled))
        if not row:
            raise NotFoundError("Model")
        return row

    @task(
        type="database",
        api=True,
        permission="llm:provider_manage",
        name="set_model_name",
        description="Задать кастомное имя модели",
        args={"model_id": "str", "display_name": "str"},
        return_type="dict",
    )
    async def set_model_name(
        self,
        model_id: str,
        display_name: str,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        repo = self._providers_repo(_session_user_id)
        name = (display_name or "").strip()
        if not name:
            raise LLMError("display_name required", "INVALID_NAME")
        row = await repo.set_model_name(model_id, name)
        if not row:
            raise NotFoundError("Model")
        return row

    @task(
        type="database",
        api=True,
        permission="llm:provider_manage",
        name="set_provider_models_enabled",
        description="Включить или выключить все модели провайдера",
        args={"provider_id": "str", "enabled": "bool"},
        return_type="dict",
    )
    async def set_provider_models_enabled(
        self,
        provider_id: str,
        enabled: bool,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        repo = self._providers_repo(_session_user_id)
        count = await repo.set_provider_models_enabled(provider_id, bool(enabled))
        return {"ok": True, "count": count, "enabled": bool(enabled)}

    @task(
        type="database",
        api=True,
        permission="llm:provider_manage",
        name="set_model_reasoning",
        description="Включить reasoning и задать effort у модели",
        args={"model_id": "str", "reasoning_enabled": "bool", "reasoning_effort": "str"},
        return_type="dict",
    )
    async def set_model_reasoning(
        self,
        model_id: str,
        reasoning_enabled: bool,
        reasoning_effort: str | None = None,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        repo = self._providers_repo(_session_user_id)
        effort = (reasoning_effort or "medium").strip().lower()
        if effort not in {"none", "low", "medium", "high"}:
            effort = "medium"
        row = await repo.set_model_reasoning(model_id, bool(reasoning_enabled), effort)
        if not row:
            raise NotFoundError("Model")
        return row

    @task(
        type="database",
        api=True,
        permission="llm:provider_share",
        name="share_provider",
        description="Расшарить своего провайдера на группу",
        args={"provider_id": "str", "group_id": "str"},
        return_type="dict",
    )
    async def share_provider(
        self,
        provider_id: str,
        group_id: str,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        if not _session_user_id or self._system_repo is None:
            raise LLMError("Authentication required", "AUTH_ERROR")
        row = await self._system_repo.get_provider(provider_id)
        if not row:
            raise LLMError(
                "share requires organization provider",
                "FORBIDDEN",
                human="Share organization providers only",
            )
        share = await self._system_repo.insert_share(str(_session_user_id), provider_id, group_id)
        return {"ok": True, "share": share}

    @task(
        type="database",
        api=True,
        permission="llm:provider_share",
        name="unshare_provider",
        description="Убрать шаринг провайдера с группы",
        args={"provider_id": "str", "group_id": "str"},
        return_type="dict",
    )
    async def unshare_provider(
        self,
        provider_id: str,
        group_id: str,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        if not _session_user_id or self._system_repo is None:
            raise LLMError("Authentication required", "AUTH_ERROR")
        ok = await self._system_repo.delete_share(provider_id, group_id)
        if not ok:
            raise NotFoundError("Share")
        return {"ok": True}

    @task(
        type="database",
        api=True,
        permission="llm:provider_manage",
        name="list_provider_shares",
        description="Группы, на которые расшарен провайдер",
        args={"provider_id": "str"},
        return_type="dict",
    )
    async def list_provider_shares(
        self,
        provider_id: str,
        _session_user_id: str | None = None,
    ) -> dict[str, Any]:
        if not _session_user_id or self._system_repo is None:
            return {"items": []}
        items = await self._system_repo.list_shares_for_provider(provider_id)
        return {"items": items}

    async def _shared_providers(self, user_id: str, mine_ids: set[str]) -> list[dict[str, Any]]:
        from modules.auth.provider import AuthProvider

        auth = self._state.services.resolve(AuthProvider)
        groups = await auth.get_user_groups(user_id)
        group_ids = [str(item["id"]) for item in groups]
        everyone = await self._system_repo.everyone_group_id()
        if everyone:
            group_ids.append(everyone)
        shares = await self._system_repo.list_shares_for_groups(group_ids)
        extra: list[dict[str, Any]] = []
        seen = set(mine_ids)
        for share in shares:
            pid = str(share.get("provider_id"))
            if pid in seen:
                continue
            raw = await self._system_repo.get_provider(pid)
            if not raw:
                continue
            public = self._system_repo._public_provider(raw)
            public["models"] = await self._system_repo.list_models(pid)
            public["owned"] = False
            public["shared"] = True
            public["common"] = True
            seen.add(pid)
            extra.append(public)
        return extra

    async def _fetch_remote_models(self, base_url: str, api_key: str) -> list[dict[str, Any]]:
        import httpx

        url = _models_endpoint(base_url)
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
        except httpx.RequestError as exc:
            raise LLMError("Wrong URL", "WRONG_URL") from exc
        if response.status_code >= 400:
            raise LLMError("Wrong URL", "WRONG_URL")
        try:
            payload = response.json()
        except Exception as exc:
            raise LLMError("Wrong URL", "WRONG_URL") from exc
        return _parse_models(payload)


def _cipher_key(api_key: str | None) -> str | None:
    if not api_key:
        return None
    try:
        return encrypt_secret(api_key)
    except Exception as exc:
        raise LLMError("secrets key is not configured", "SECRETS") from exc


def _validate_base_url(base_url: str) -> str:
    from urllib.parse import urlparse

    raw = (base_url or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMError("Wrong URL", "WRONG_URL")
    return raw


def _models_endpoint(base_url: str) -> str:
    url = _validate_base_url(base_url).rstrip("/")
    if url.endswith("/models"):
        return url
    return f"{url}/models"


def _parse_models(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw = payload.get("data") or payload.get("models") or payload.get("items") or []
        if not raw:
            for value in payload.values():
                if isinstance(value, list) and value:
                    raw = value
                    break
    elif isinstance(payload, list):
        raw = payload
    else:
        raw = []
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        extra: dict[str, Any] | None
        if isinstance(item, str):
            mid = item
            extra = None
        elif isinstance(item, dict):
            mid = str(item.get("id") or item.get("name") or "")
            extra = item
        else:
            mid = ""
            extra = None
        if not mid or mid in seen:
            continue
        seen.add(mid)
        items.append(
            {
                "id": mid,
                "name": mid,
                "supports_reasoning": _supports_reasoning(mid, extra),
            }
        )
    return items


def _supports_reasoning(model_id: str, extra: dict[str, Any] | None) -> bool:
    if extra:
        if extra.get("supports_reasoning") is True or extra.get("reasoning") is True:
            return True
        caps = extra.get("capabilities")
        if isinstance(caps, dict) and (caps.get("reasoning") or caps.get("thinking")):
            return True
    name = model_id.lower()
    markers = (
        "reasoning",
        "think",
        "-r1",
        "o1",
        "o3",
        "o4",
        "gpt-5",
        "grok-3-mini",
        "grok-4",
    )
    return any(marker in name for marker in markers)
