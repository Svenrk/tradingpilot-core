from __future__ import annotations

from collections.abc import Mapping
from logging.config import fileConfig
from typing import Protocol


class AlembicConfigWithLogging(Protocol):
    config_file_name: str | None

    def get_section(
        self, name: str, default: Mapping[str, str] | None = None
    ) -> Mapping[str, str] | None: ...


def configure_alembic_logging(config: AlembicConfigWithLogging) -> bool:
    if config.config_file_name is None:
        return False
    if any(config.get_section(section) is None for section in ("loggers", "handlers", "formatters")):
        return False
    fileConfig(config.config_file_name)
    return True
