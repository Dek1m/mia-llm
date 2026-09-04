"""OAuth-вендоры LLM. Сейчас device-code для xAI; остальные — в тот же реестр."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from .secrets import decrypt_secret, encrypt_secret

__all__ = [
    "OAuthVendor",
    "list_vendors",
    "get_vendor",
    "request_device_code",
    "poll_device_token",
    "pack_device",
    "pack_tokens",
    "unpack_secret",
    "bearer_from_stored",
    "is_expired",
    "refresh_access_token",
]

_REFRESH_SKEW = 120


@dataclass(frozen=True)
class OAuthVendor:
    id: str
    name: str
    base_url: str
    client_id: str
    scope: str
    device_url: str
    token_url: str


# Публичный client_id Grok CLI (секрета нет). Device-code RFC 8628.
_VENDORS: dict[str, OAuthVendor] = {
    "xai": OAuthVendor(
        id="xai",
        name="xAI",
        base_url="https://api.x.ai/v1",
        client_id="b1a00492-073a-47ea-816f-4c329264a828",
        scope="openid profile email offline_access grok-cli:access api:access",
        device_url="https://auth.x.ai/oauth2/device/code",
        token_url="https://auth.x.ai/oauth2/token",
    ),
}


def list_vendors() -> list[dict[str, str]]:
    return [{"id": item.id, "name": item.name, "mode": "device_code"} for item in _VENDORS.values()]


def get_vendor(vendor_id: str) -> OAuthVendor | None:
    return _VENDORS.get((vendor_id or "").strip().lower())


def pack_device(device_code: str, interval: int, expires_at: int) -> str:
    return encrypt_secret(
        json.dumps(
            {
                "kind": "device",
                "device_code": device_code,
                "interval": interval,
                "expires_at": expires_at,
            }
        )
    ) or ""


def pack_tokens(access: str, refresh: str | None, expires_at: int) -> str:
    return encrypt_secret(
        json.dumps(
            {
                "kind": "token",
                "access": access,
                "refresh": refresh or "",
                "expires_at": expires_at,
            }
        )
    ) or ""


def unpack_secret(stored: str | None) -> dict[str, Any] | None:
    raw = decrypt_secret(stored)
    if not raw:
        return None
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict):
            return data
        return None
    return {"kind": "token", "access": raw, "refresh": "", "expires_at": 0}


def bearer_from_stored(stored: str | None) -> str | None:
    data = unpack_secret(stored)
    if not data or data.get("kind") != "token":
        return None
    access = str(data.get("access") or "").strip()
    return access or None


def is_expired(data: dict[str, Any] | None, now: int | None = None, skew: int = _REFRESH_SKEW) -> bool:
    """True если access-токен просрочен или истечёт в ближайшие skew секунд.
    expires_at=0 — сырой API-ключ, не истекает."""
    if not data or data.get("kind") != "token":
        return False
    expires_at = int(data.get("expires_at") or 0)
    if expires_at <= 0:
        return False
    stamp = int(now if now is not None else time.time())
    return expires_at <= stamp + skew


async def request_device_code(vendor: OAuthVendor) -> dict[str, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            vendor.device_url,
            data={"client_id": vendor.client_id, "scope": vendor.scope},
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
    payload = _json_body(response)
    if response.status_code >= 400:
        raise RuntimeError(str(payload.get("error_description") or payload.get("error") or "oauth failed"))
    device_code = str(payload.get("device_code") or "")
    user_code = str(payload.get("user_code") or "")
    if not device_code or not user_code:
        raise RuntimeError("oauth device response incomplete")
    interval = int(payload.get("interval") or 5)
    expires_in = int(payload.get("expires_in") or 900)
    uri = str(payload.get("verification_uri") or payload.get("verification_uri_complete") or "")
    complete = str(payload.get("verification_uri_complete") or uri)
    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": uri,
        "verification_uri_complete": complete,
        "interval": max(interval, 3),
        "expires_in": expires_in,
        "expires_at": int(time.time()) + expires_in,
    }


async def poll_device_token(vendor: OAuthVendor, device_code: str) -> dict[str, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            vendor.token_url,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code,
                "client_id": vendor.client_id,
            },
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
    payload = _json_body(response)
    error = str(payload.get("error") or "")
    if error == "authorization_pending":
        return {"status": "pending"}
    if error == "slow_down":
        return {"status": "slow_down"}
    if error in {"expired_token", "expired"}:
        return {"status": "expired"}
    if error in {"access_denied", "authorization_denied"}:
        return {"status": "denied"}
    access = str(payload.get("access_token") or "")
    if not access:
        if response.status_code >= 400:
            return {"status": "error", "detail": error or f"http_{response.status_code}"}
        return {"status": "pending"}
    expires_in = int(payload.get("expires_in") or 21600)
    return {
        "status": "connected",
        "access": access,
        "refresh": str(payload.get("refresh_token") or ""),
        "expires_at": int(time.time()) + expires_in,
    }


async def refresh_access_token(vendor: OAuthVendor, refresh_token: str) -> dict[str, Any]:
    """RFC 6749 refresh_token. xAI access живёт часы — без этого chat даёт 403."""
    import httpx

    token = (refresh_token or "").strip()
    if not token:
        raise RuntimeError("oauth refresh missing")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            vendor.token_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": token,
                "client_id": vendor.client_id,
            },
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
    payload = _json_body(response)
    access = str(payload.get("access_token") or "")
    if not access:
        detail = str(payload.get("error_description") or payload.get("error") or f"http_{response.status_code}")
        raise RuntimeError(detail)
    expires_in = int(payload.get("expires_in") or 21600)
    return {
        "access": access,
        "refresh": str(payload.get("refresh_token") or token),
        "expires_at": int(time.time()) + expires_in,
    }


def _json_body(response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
