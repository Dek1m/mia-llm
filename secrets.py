"""Шифрование API-ключей провайдеров. В БД только ciphertext."""
from __future__ import annotations

import base64
import hashlib
import os

PREFIX = "v1:"

__all__ = ["encrypt_secret", "decrypt_secret", "is_encrypted"]


def _key() -> bytes:
    raw = os.getenv("LLM_SECRETS_KEY") or os.getenv("AUTH_JWT_SECRET") or ""
    if not raw:
        raise RuntimeError("LLM_SECRETS_KEY or AUTH_JWT_SECRET is required")
    return hashlib.sha256(raw.encode("utf-8")).digest()


def is_encrypted(value: str | None) -> bool:
    return bool(value) and value.startswith(PREFIX)


def encrypt_secret(plain: str | None) -> str | None:
    if not plain:
        return plain
    if is_encrypted(plain):
        return plain
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(12)
    token = nonce + AESGCM(_key()).encrypt(nonce, plain.encode("utf-8"), None)
    return PREFIX + base64.urlsafe_b64encode(token).decode("ascii")


def decrypt_secret(stored: str | None) -> str | None:
    if not stored:
        return stored
    if not is_encrypted(stored):
        return stored
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    raw = base64.urlsafe_b64decode(stored[len(PREFIX):].encode("ascii"))
    return AESGCM(_key()).decrypt(raw[:12], raw[12:], None).decode("utf-8")
