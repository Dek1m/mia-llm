"""_validate_base_url: схема + SSRF-guard приватных/служебных адресов."""
from __future__ import annotations

import socket

import pytest

from modules.llm.provider import LLMError, _validate_base_url


@pytest.fixture(autouse=True)
def _no_private_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_ALLOW_PRIVATE_BASE", raising=False)


def _addrinfo(ip: str):
    def resolve(host: str, port: int | None, *args, **kwargs):
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    return resolve


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:11434/v1",
        "http://10.0.0.5/v1",
        "http://172.16.0.1/v1",
        "http://192.168.1.1/v1",
        "http://0.0.0.0/v1",
        "http://[::1]/v1",
        "http://[fd00::1]/v1",
        "http://[fe80::1]/v1",
    ],
)
def test_private_ip_literal_rejected(url: str) -> None:
    """IP-литералы приватных/служебных диапазонов — WRONG_URL, без DNS."""
    with pytest.raises(LLMError) as exc_info:
        _validate_base_url(url)
    assert exc_info.value.code == "WRONG_URL"


def test_hostname_resolving_to_private_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """DNS-ребро SSRF: hostname → 127.0.0.1 отклоняется."""
    monkeypatch.setattr(socket, "getaddrinfo", _addrinfo("127.0.0.1"))
    with pytest.raises(LLMError):
        _validate_base_url("http://llama.internal/v1")


def test_public_https_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """https://api.z.ai/... с публичным IP проходит (DNS замокан — оффлайн)."""
    monkeypatch.setattr(socket, "getaddrinfo", _addrinfo("93.184.216.34"))
    url = "https://api.z.ai/api/paas/v4"
    assert _validate_base_url(url) == url


def test_dns_failure_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(host, port, *args, **kwargs):
        raise socket.gaierror("name resolution failed")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    with pytest.raises(LLMError) as exc_info:
        _validate_base_url("https://no-such-host.invalid/v1")
    assert exc_info.value.code == "WRONG_URL"


def test_env_flag_allows_private(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM_ALLOW_PRIVATE_BASE=true — локальная llama.cpp разрешена."""
    monkeypatch.setenv("LLM_ALLOW_PRIVATE_BASE", "true")
    url = "http://127.0.0.1:11434/v1"
    assert _validate_base_url(url) == url


@pytest.mark.parametrize("url", ["ftp://api.z.ai/v1", "not-a-url", "https://"])
def test_bad_scheme_or_netloc_rejected(url: str) -> None:
    with pytest.raises(LLMError) as exc_info:
        _validate_base_url(url)
    assert exc_info.value.code == "WRONG_URL"
