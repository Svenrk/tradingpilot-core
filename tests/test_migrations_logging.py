from __future__ import annotations

from collections.abc import Mapping

from migrations.logging import configure_alembic_logging


class DummyConfig:
    def __init__(self, config_file_name: str | None, sections: dict[str, Mapping[str, str] | None]) -> None:
        self.config_file_name = config_file_name
        self._sections = sections

    def get_section(
        self, name: str, default: Mapping[str, str] | None = None
    ) -> Mapping[str, str] | None:
        return self._sections.get(name, default)


def test_configure_alembic_logging_skips_missing_sections(monkeypatch) -> None:
    called = False

    def fake_file_config(filename: str) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("migrations.logging.fileConfig", fake_file_config)

    configured = configure_alembic_logging(
        DummyConfig("alembic.ini", {"loggers": {}, "handlers": None, "formatters": None})
    )

    assert configured is False
    assert called is False


def test_configure_alembic_logging_loads_complete_config(monkeypatch) -> None:
    called_with: list[str] = []

    def fake_file_config(filename: str) -> None:
        called_with.append(filename)

    monkeypatch.setattr("migrations.logging.fileConfig", fake_file_config)

    configured = configure_alembic_logging(
        DummyConfig("alembic.ini", {"loggers": {}, "handlers": {}, "formatters": {}})
    )

    assert configured is True
    assert called_with == ["alembic.ini"]
