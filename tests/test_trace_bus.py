from __future__ import annotations

from modules.llm.trace_bus import clear_trace, get_trace, put_trace


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def delete(self, key: str) -> None:
        self.store.pop(key, None)


def test_put_get_clear() -> None:
    bus = _FakeRedis()
    put_trace("s1", {"content": "hi", "reasoning": "", "stages": []}, client=bus)
    got = get_trace("s1", client=bus)
    assert got is not None
    assert got["content"] == "hi"
    clear_trace("s1", client=bus)
    assert get_trace("s1", client=bus) is None
