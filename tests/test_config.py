"""R14: typed config via pydantic-settings, .env.example, no secrets in repo."""

from __future__ import annotations

from pathlib import Path

import pytest

from geo_evidence_mcp.config import Settings
from geo_evidence_mcp.init_env import PLACEHOLDER, render_env

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "ANTHROPIC_API_KEY",
        "POSTGRES_PASSWORD",
        "STAC_ENDPOINT",
        "LANGFUSE_SECRET_KEY",
        "DB_PORT",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.req("R14")
def test_defaults_load_without_dotenv() -> None:
    s = Settings(_env_file=None)
    assert s.langfuse_host == "http://localhost:3000"
    assert s.db_port == 5432
    assert s.anthropic_api_key is None


@pytest.mark.req("R14")
def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAC_ENDPOINT", "https://example.test/stac")
    monkeypatch.setenv("DB_PORT", "6543")
    s = Settings(_env_file=None)
    assert s.stac_endpoint == "https://example.test/stac"
    assert s.db_port == 6543


@pytest.mark.req("R14")
def test_secrets_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "lf-secret-not-real")
    s = Settings(_env_file=None)
    assert "sk-test-not-real" not in repr(s)
    assert "lf-secret-not-real" not in repr(s)


@pytest.mark.req("R14")
def test_database_url_requires_and_quotes_password(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
        _ = Settings(_env_file=None).database_url
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss/word")
    url = Settings(_env_file=None).database_url
    assert url == "postgresql://gem:p%40ss%2Fword@localhost:5432/gem"


@pytest.mark.req("R14")
def test_render_env_generates_unique_secrets_and_mirrors_langfuse_keys() -> None:
    template = (ROOT / ".env.example").read_text(encoding="utf-8")
    values = dict(
        line.split("=", 1)
        for line in render_env(template).splitlines()
        if line and not line.startswith("#")
    )
    assert PLACEHOLDER not in values.values()
    assert len(values["ENCRYPTION_KEY"]) == 64
    assert values["LANGFUSE_PUBLIC_KEY"] == values["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"]
    assert values["LANGFUSE_SECRET_KEY"] == values["LANGFUSE_INIT_PROJECT_SECRET_KEY"]
    assert values["LANGFUSE_PUBLIC_KEY"].startswith("pk-lf-")
    assert values["NEXTAUTH_SECRET"] != values["SALT"]


@pytest.mark.req("R14")
def test_env_example_contains_no_real_looking_secrets() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "sk-ant-" not in text
    for line in text.splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            assert line == "ANTHROPIC_API_KEY="
