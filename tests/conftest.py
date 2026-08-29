"""LLM Tests — конфигурация и фикстуры (MockPool + фейковый провайдер)."""
from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest


# ── Динамическая загрузка модуля llm ──────────────────

_LLM_DIR = Path(__file__).resolve().parent.parent

# Регистрируем modules.llm (если ещё нет)
if "modules.llm" not in sys.modules:
    _fake_pkg = types.ModuleType("modules.llm")
    _fake_pkg.__path__ = [str(_LLM_DIR)]
    _fake_pkg.__package__ = "modules.llm"
    sys.modules["modules.llm"] = _fake_pkg

    for submod in ["config", "schema", "schemas", "repository", "models"]:
        file_path = _LLM_DIR / f"{submod}.py"
        if file_path.exists():
            spec = importlib.util.spec_from_file_location(
                f"modules.llm.{submod}", file_path,
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = "modules.llm"
                sys.modules[f"modules.llm.{submod}"] = mod
                spec.loader.exec_module(mod)
                setattr(_fake_pkg, submod, mod)

    # Providers subpackage
    providers_dir = _LLM_DIR / "providers"
    if providers_dir.exists() and "modules.llm.providers" not in sys.modules:
        _fake_providers = types.ModuleType("modules.llm.providers")
        _fake_providers.__path__ = [str(providers_dir)]
        _fake_providers.__package__ = "modules.llm.providers"
        sys.modules["modules.llm.providers"] = _fake_providers
        setattr(_fake_pkg, "providers", _fake_providers)

        for submod in ["base", "openai", "registry"]:
            file_path = providers_dir / f"{submod}.py"
            if file_path.exists():
                spec = importlib.util.spec_from_file_location(
                    f"modules.llm.providers.{submod}", file_path,
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    mod.__package__ = "modules.llm.providers"
                    sys.modules[f"modules.llm.providers.{submod}"] = mod
                    spec.loader.exec_module(mod)
                    setattr(_fake_providers, submod, mod)

# Загружаем provider.py (может не быть в корневом conftest из-за auth dep)
if "modules.llm.provider" not in sys.modules:
    _provider_file = _LLM_DIR / "provider.py"
    if _provider_file.exists():
        _spec = importlib.util.spec_from_file_location(
            "modules.llm.provider", _provider_file,
        )
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            _mod.__package__ = "modules.llm"
            sys.modules["modules.llm.provider"] = _mod
            _spec.loader.exec_module(_mod)
            _fake_pkg_ref = sys.modules.get("modules.llm")
            if _fake_pkg_ref:
                setattr(_fake_pkg_ref, "provider", _mod)


# ── MockPool ─────────────────────────────────────────

class _MockRow:
    """Строка из БД — dict-like."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self) -> str:
        return f"_MockRow({self._data})"


class MockPool:
    """Фейковый пул соединений для тестов LLM-модуля.

    Хранит данные в памяти, имитирует fetchrow/fetch/fetchval/execute.
    """

    def __init__(self) -> None:
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._executed: list[tuple[str, tuple]] = []

    def _ensure_table(self, table: str) -> list[dict[str, Any]]:
        if table not in self._tables:
            self._tables[table] = []
        return self._tables[table]

    async def fetchrow(self, query: str, *args: Any) -> _MockRow | None:
        self._executed.append((query, args))
        table = self._detect_table(query)

        # INSERT ... RETURNING *
        if "INSERT" in query and "RETURNING" in query:
            return self._handle_insert(table, query, args)

        # UPDATE ... RETURNING *
        if "UPDATE" in query and "RETURNING" in query:
            return self._handle_update(table, query, args)

        # SELECT ... WHERE id = $1
        if "SELECT" in query and "WHERE id =" in query:
            rows = self._ensure_table(table)
            for row in rows:
                if str(row.get("id")) == str(args[0]):
                    return _MockRow(row)
            return None

        # SELECT ... WHERE name = $1
        if "SELECT" in query and "WHERE name =" in query:
            rows = self._ensure_table(table)
            for row in rows:
                if row.get("name") == args[0]:
                    return _MockRow(row)
            return None

        return None

    async def fetchval(self, query: str, *args: Any) -> Any:
        self._executed.append((query, args))
        table = self._detect_table(query)

        if "COUNT(*)" in query:
            rows = self._ensure_table(table)
            if "WHERE agent_type" in query:
                return sum(1 for r in rows if r.get("agent_type") == args[0])
            return len(rows)

        return None

    async def fetch(self, query: str, *args: Any) -> list[_MockRow]:
        self._executed.append((query, args))
        table = self._detect_table(query)
        rows = self._ensure_table(table)

        if "WHERE provider_id" in query:
            filtered = [r for r in rows if str(r.get("provider_id")) == str(args[0])]
            return [_MockRow(r) for r in filtered]

        # SELECT ... ORDER BY ... LIMIT ... OFFSET
        if "ORDER BY" in query:
            filtered = rows
            if "WHERE agent_type" in query:
                filtered = [r for r in rows if r.get("agent_type") == args[0]]

            # Parse LIMIT and OFFSET from args
            limit = args[-2] if len(args) >= 2 else 100
            offset = args[-1] if len(args) >= 1 else 0
            return [_MockRow(r) for r in filtered[offset:offset + limit]]

        return [_MockRow(r) for r in rows]

    async def execute(self, query: str, *args: Any) -> str:
        self._executed.append((query, args))
        table = self._detect_table(query)

        # DELETE
        if "DELETE" in query and "WHERE id =" in query:
            rows = self._ensure_table(table)
            before = len(rows)
            self._tables[table] = [r for r in rows if str(r.get("id")) != str(args[0])]
            deleted = before - len(self._tables[table])
            return f"DELETE {deleted}"

        return "OK"

    def _detect_table(self, query: str) -> str:
        """Определяет имя таблицы из запроса."""
        for prefix in ["llm.", ""]:
            for table in ["llm_agents", "llm_providers", "llm_models"]:
                full = f"{prefix}{table}"
                if full in query:
                    return full
        return "unknown"

    def _handle_insert(self, table: str, query: str, args: tuple) -> _MockRow:
        """Обработка INSERT ... RETURNING *."""
        rows = self._ensure_table(table)

        if "llm_models" in table:
            new_id = str(uuid.uuid4())
            new_row = {
                "id": new_id,
                "provider_id": args[0],
                "model_id": args[1],
                "display_name": args[2],
                "enabled": args[3] if len(args) > 3 else True,
                "is_available": True,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            rows.append(new_row)
            return _MockRow(new_row)

        if "llm_providers" in table:
            new_id = str(uuid.uuid4())
            new_row = {
                "id": new_id,
                "name": args[0],
                "kind": args[1],
                "vendor": args[2],
                "description": args[3] if len(args) > 3 else None,
                "base_url": args[4] if len(args) > 4 else None,
                "default_model": args[5] if len(args) > 5 else None,
                "api_key": args[6] if len(args) > 6 else None,
                "oauth_status": args[7] if len(args) > 7 else None,
                "is_active": True,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            rows.append(new_row)
            return _MockRow(new_row)

        if "ON CONFLICT" in query:
            name = args[0]
            for row in rows:
                if row.get("name") == name:
                    row["description"] = args[2] or row.get("description")
                    row["system_prompt"] = args[3] or row.get("system_prompt")
                    row["model"] = args[4] or row.get("model")
                    row["settings"] = json.loads(args[5]) if args[5] else row.get("settings", {})
                    row["updated_at"] = datetime.now(timezone.utc)
                    return _MockRow(row)

        new_id = str(uuid.uuid4())
        new_row = {
            "id": new_id,
            "name": args[0],
            "agent_type": args[1],
            "description": args[2],
            "system_prompt": args[3],
            "model": args[4],
            "settings": json.loads(args[5]) if args[5] else {},
            "workspace_id": args[6],
            "owner_id": args[7],
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        rows.append(new_row)
        return _MockRow(new_row)

    def _handle_update(self, table: str, query: str, args: tuple) -> _MockRow | None:
        """Обработка UPDATE ... RETURNING *."""
        rows = self._ensure_table(table)
        agent_id = args[-1]

        set_match = re.search(r"SET\s+(.+?)\s+WHERE", query)
        if not set_match:
            return None

        set_clause = set_match.group(1)
        column_updates: dict[str, Any] = {}

        for part in set_clause.split(","):
            part = part.strip()
            if "NOW()" in part:
                continue
            eq_match = re.match(r"(\w+)\s*=\s*\$(\d+)", part)
            if eq_match:
                col_name = eq_match.group(1)
                param_idx = int(eq_match.group(2)) - 1
                if param_idx < len(args):
                    column_updates[col_name] = args[param_idx]

        for row in rows:
            if str(row.get("id")) == str(agent_id):
                row.update(column_updates)
                row["updated_at"] = datetime.now(timezone.utc)
                return _MockRow(row)
        return None


# ── Фикстуры ─────────────────────────────────────────

@pytest.fixture
def mock_pool() -> MockPool:
    """Фейковый пул соединений."""
    return MockPool()


class FakeLLMProvider:
    """Фейковый LLM-провайдер для тестов."""

    def __init__(self, name: str = "fake", should_fail: bool = False) -> None:
        self._name = name
        self._should_fail = should_fail

    @property
    def name(self) -> str:
        return self._name

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        if self._should_fail:
            raise ConnectionError("Fake provider connection error")
        return {
            "content": f"[fake:{self._name}] Echo",
            "role": "assistant",
            "model": "fake-model",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def models(self) -> list[str]:
        return ["fake-model", "fake-model-2"]

    async def health(self) -> bool:
        return not self._should_fail


@pytest.fixture
def fake_llm_provider() -> FakeLLMProvider:
    """Фейковый LLM-провайдер."""
    return FakeLLMProvider()
