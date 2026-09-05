from __future__ import annotations

from modules.llm.trace_bus import (
    add_chat_tokens,
    clear_trace,
    get_chat_tokens,
    get_trace,
    put_trace,
)


class _FakePipeline:
    def __init__(self, store: dict[str, dict[str, int]]) -> None:
        self._store = store
        self._ops: list[tuple] = []

    def hincrby(self, key: str, field: str, amount: int) -> None:
        self._ops.append(("hincrby", key, field, amount))

    def expire(self, key: str, ttl: int) -> None:
        self._ops.append(("expire", key, ttl))

    def execute(self) -> None:
        for op in self._ops:
            if op[0] == "hincrby":
                _, key, field, amount = op
                bucket = self._store.setdefault(key, {})
                bucket[field] = bucket.get(field, 0) + amount
        self._ops.clear()


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.hashes: dict[str, dict[str, int]] = {}

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def delete(self, key: str) -> None:
        self.store.pop(key, None)

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self.hashes)

    def hgetall(self, key: str) -> dict[str, int]:
        return self.hashes.get(key, {})


def test_put_get_clear() -> None:
    bus = _FakeRedis()
    put_trace("u1", "s1", {"content": "hi", "reasoning": "", "stages": []}, client=bus)
    got = get_trace("u1", "s1", client=bus)
    assert got is not None
    assert got["content"] == "hi"
    clear_trace("u1", "s1", client=bus)
    assert get_trace("u1", "s1", client=bus) is None


def test_keys_contain_user_id() -> None:
    bus = _FakeRedis()
    put_trace("u1", "s1", {"content": "x"}, client=bus)
    assert "llm:trace:u1:s1" in bus.store
    add_chat_tokens("u1", "s1", 3, 4, client=bus)
    assert "llm:chat_stats:u1:s1" in bus.hashes


def test_other_user_cannot_read_trace() -> None:
    bus = _FakeRedis()
    put_trace("u1", "s1", {"content": "secret"}, client=bus)
    assert get_trace("u2", "s1", client=bus) is None
    # clear чужим user_id не трогает трассу владельца
    clear_trace("u2", "s1", client=bus)
    assert get_trace("u1", "s1", client=bus) is not None


def test_chat_tokens_scoped_and_accumulated() -> None:
    bus = _FakeRedis()
    add_chat_tokens("u1", "s1", 10, 20, client=bus)
    add_chat_tokens("u1", "s1", 5, 0, client=bus)
    assert get_chat_tokens("u1", "s1", client=bus) == {"in": 15, "out": 20}
    assert get_chat_tokens("u2", "s1", client=bus) == {"in": 0, "out": 0}


def test_empty_user_id_is_noop() -> None:
    bus = _FakeRedis()
    put_trace("", "s1", {"content": "x"}, client=bus)
    assert bus.store == {}
    assert get_trace("", "s1", client=bus) is None
    assert get_chat_tokens("", "s1", client=bus) == {"in": 0, "out": 0}
