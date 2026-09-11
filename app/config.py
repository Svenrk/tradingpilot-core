from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODULES = [
    "tradingview",
    "market_data",
    "signal_engine",
    "risk_engine",
    "execution",
    "positions",
    "auth",
    "settings",
    "dashboard",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="TP_", extra="ignore")

    app_name: str = "TradingPilot Core"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite+aiosqlite:///./tradingpilot.db"
    redis_url: str | None = None
    enabled_modules: list[str] = Field(default_factory=lambda: list(DEFAULT_MODULES))
    symbol_allowlist: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframe_allowlist: list[str] = Field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    tv_webhook_secret: str = "change-me"
    session_cookie_name: str = "tp_session"
    session_cookie_secure: bool = False
    session_ttl_seconds: int = 60 * 60 * 12
    enable_live_trading: bool = False
    default_broker: str = "paper"
    admin_username: str = "admin"
    admin_password: str = "change-me-now"
    admin_password_hash: str | None = None
    admin_role: str = "paper"
    log_level: str = "INFO"
    market_data_cache_ttl_seconds: float = 5.0
    latest_price_ttl_seconds: int = 300
    worker_concurrency: int = 4
    worker_batch_size: int = 10
    pending_claim_idle_seconds: int = 60
    retention_days: int = 30
    retention_interval_seconds: float = 3600.0
    metrics_interval_seconds: float = 60.0
    login_max_attempts: int = 5
    login_attempt_window_seconds: int = 300

    @field_validator("enabled_modules", "symbol_allowlist", "timeframe_allowlist", mode="before")
    @classmethod
    def _split_csv(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
