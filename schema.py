"""LLM AUTH_SCHEMA — permissions и roles для модуля llm.

Регистрация через AuthSchemaRegistry.register("llm", LLM_SCHEMA).
"""
from __future__ import annotations

from typing import Any

__all__ = ["LLM_SCHEMA"]

LLM_SCHEMA: dict[str, list[dict[str, Any]]] = {
    "permissions": [
        {"name": "llm:chat", "description": "Вызов LLM-провайдеров (chat completions)"},
        {"name": "llm:chat_stream", "description": "Потоковый вывод LLM (SSE)"},
        {"name": "llm:agent_manage", "description": "Создание/обновление/удаление агентов"},
        {"name": "llm:agent_list", "description": "Просмотр списка агентов и их информации"},
        {"name": "llm:config", "description": "Просмотр конфигурации провайдеров и моделей"},
    ],
    "roles": [
        {
            "name": "llm_operator",
            "description": "Оператор LLM: вызов чата и просмотр агентов",
            "permissions": [
                "llm:chat",
                "llm:chat_stream",
                "llm:agent_list",
            ],
        },
        {
            "name": "llm_admin",
            "description": "Полный контроль над LLM-модулем",
            "permissions": [
                "llm:*",
            ],
        },
    ],
}
