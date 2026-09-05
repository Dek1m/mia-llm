"""secrets: roundtrip и WARN при фолбэке на AUTH_JWT_SECRET (один раз)."""
from __future__ import annotations

import logging

import pytest

import modules.llm.secrets as secrets


def test_roundtrip_with_dedicated_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_SECRETS_KEY", "test-llm-secrets")
    stored = secrets.encrypt_secret("sk-secret")
    assert stored is not None and stored.startswith(secrets.PREFIX)
    assert secrets.decrypt_secret(stored) == "sk-secret"


def test_fallback_to_jwt_secret_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("LLM_SECRETS_KEY", raising=False)
    monkeypatch.setenv("AUTH_JWT_SECRET", "jwt-fallback-key")
    monkeypatch.setattr(secrets, "_fallback_warned", False)
    with caplog.at_level(logging.WARNING, logger="modules.llm.secrets"):
        stored = secrets.encrypt_secret("sk-secret")
        secrets.encrypt_secret("sk-secret-2")
    assert secrets.decrypt_secret(stored) == "sk-secret"
    warnings = [r for r in caplog.records if "llm_secrets_key_fallback" in r.getMessage()]
    assert len(warnings) == 1


def test_no_keys_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_SECRETS_KEY", raising=False)
    monkeypatch.delenv("AUTH_JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="LLM_SECRETS_KEY or AUTH_JWT_SECRET"):
        secrets.encrypt_secret("sk-secret")
