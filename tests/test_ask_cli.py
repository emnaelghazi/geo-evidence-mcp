"""R2: the gem-ask command line refuses cleanly without credentials."""

from __future__ import annotations

import pytest

from geo_evidence_mcp.agent.ask import main
from geo_evidence_mcp.config import get_settings


@pytest.mark.req("R2")
def test_gem_ask_without_api_key_exits_with_a_clear_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: pytest.TempPathFactory,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(str(tmp_path))  # no .env here
    get_settings.cache_clear()
    try:
        assert main(["What is the NDVI?"]) == 2
    finally:
        get_settings.cache_clear()
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out
