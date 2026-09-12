from __future__ import annotations

import pytest

from app.config import DEFAULT_MODULES, Settings


def test_settings_parse_csv_lists_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TP_ENABLED_MODULES", "auth, markets ,dashboard")
    monkeypatch.setenv("TP_SYMBOL_ALLOWLIST", "BTCUSDT")
    monkeypatch.setenv("TP_TIMEFRAME_ALLOWLIST", "1m,5m")

    settings = Settings(_env_file=None)

    assert settings.enabled_modules == ["auth", "markets", "dashboard"]
    assert settings.symbol_allowlist == ["BTCUSDT"]
    assert settings.timeframe_allowlist == ["1m", "5m"]


def test_settings_list_defaults_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TP_ENABLED_MODULES", "TP_SYMBOL_ALLOWLIST", "TP_TIMEFRAME_ALLOWLIST"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.enabled_modules == DEFAULT_MODULES
    assert settings.symbol_allowlist == ["BTCUSDT", "ETHUSDT"]
