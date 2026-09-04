"""OAuth helpers: expiry and refresh_token grant."""
from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from modules.llm.oauth import (
    get_vendor,
    is_expired,
    pack_tokens,
    refresh_access_token,
    unpack_secret,
)


def test_is_expired_raw_key() -> None:
    assert is_expired({"kind": "token", "access": "xai-key", "refresh": "", "expires_at": 0}) is False


def test_is_expired_future() -> None:
    now = 1_700_000_000
    assert is_expired({"kind": "token", "expires_at": now + 3600}, now=now) is False


def test_is_expired_within_skew() -> None:
    now = 1_700_000_000
    assert is_expired({"kind": "token", "expires_at": now + 30}, now=now) is True


def test_is_expired_past() -> None:
    now = 1_700_000_000
    assert is_expired({"kind": "token", "expires_at": now - 1}, now=now) is True


def test_pack_roundtrip() -> None:
    blob = pack_tokens("acc", "ref", 123)
    data = unpack_secret(blob)
    assert data is not None
    assert data["kind"] == "token"
    assert data["access"] == "acc"
    assert data["refresh"] == "ref"
    assert data["expires_at"] == 123


class _FakeResp:
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    last: dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(self, url: str, data: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> _FakeResp:
        _FakeClient.last = {"url": url, "data": dict(data or {}), "headers": dict(headers or {})}
        return _FakeResp(
            200,
            {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            },
        )


@pytest.mark.asyncio
async def test_refresh_access_token_posts_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    vendor = get_vendor("xai")
    assert vendor is not None
    before = int(time.time())
    result = await refresh_access_token(vendor, "old-refresh")
    assert result["access"] == "new-access"
    assert result["refresh"] == "new-refresh"
    assert result["expires_at"] >= before + 3600
    assert _FakeClient.last["url"] == vendor.token_url
    assert _FakeClient.last["data"]["grant_type"] == "refresh_token"
    assert _FakeClient.last["data"]["refresh_token"] == "old-refresh"
    assert _FakeClient.last["data"]["client_id"] == vendor.client_id


class _FailClient(_FakeClient):
    async def post(self, url: str, data: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> _FakeResp:
        return _FakeResp(400, {"error": "invalid_grant"})


@pytest.mark.asyncio
async def test_refresh_access_token_invalid_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _FailClient)
    vendor = get_vendor("xai")
    assert vendor is not None
    with pytest.raises(RuntimeError, match="invalid_grant"):
        await refresh_access_token(vendor, "dead")
