import pytest

from app.core.registry import ModuleRegistry


def test_registry_sorts_modules_by_dependencies() -> None:
    registry = ModuleRegistry.discover(
        ["settings", "market_data", "signal_engine", "risk_engine", "execution"]
    )
    assert [module.name for module in registry.modules] == [
        "market_data",
        "settings",
        "signal_engine",
        "risk_engine",
        "execution",
    ]


def test_registry_raises_for_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="missing dependencies"):
        ModuleRegistry.discover(["execution"])
