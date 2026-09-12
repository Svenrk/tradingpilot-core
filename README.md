# TradingPilot Core

TradingPilot Core is a stripped, production-shaped FastAPI backend built around plugin-style modules. Every feature lives in `app/modules/<name>/` and is enabled or disabled through `TP_ENABLED_MODULES`.

## Architecture

```text
TradingView -> /api/v1/tv/webhook/{webhook_id}
           -> PostgreSQL tradingview_events + event_outbox
           -> Redis Stream tv_events
           -> pipeline worker (market_data -> signal_engine -> risk_engine -> execution)
           -> PostgreSQL signals/orders/positions
           -> Redis Stream ui_updates
           -> /api/v1/events|signals|orders|positions + /api/v1/ws/updates
```

## Run with Docker Compose

1. Copy `.env.example` to `.env` and adjust secrets.
2. Start the stack:
   ```bash
   docker compose up --build
   ```
3. API: `http://localhost:8000`, health: `http://localhost:8000/api/v1/health`.
4. Markets dashboard: `http://localhost:8000/dashboard` — Crypto and Stocks tabs, each with a top-50 list by 24h volume and a top-50 list by 24h % change (JSON at `/api/v1/markets/crypto` and `/api/v1/markets/stocks`).

## Add a new module

1. Create `app/modules/<name>/__init__.py`.
2. Export `module = YourModule()` from that package.
3. Implement any needed `routers()`, `models()`, `pipeline_steps()`, `background_loops()`, `on_startup()`, and `on_shutdown()` hooks.
4. Add the module name to `TP_ENABLED_MODULES`.
5. If the module needs ordering, declare `depends_on` in the module class.
6. Add focused tests for the new behavior.

## Disable a module

Remove the module name from `TP_ENABLED_MODULES`. The registry skips importing it, its routers are not mounted, and its pipeline/background hooks are not used.

## Notes

- Risk is the final gate: execution raises unless the risk decision is approved.
- Live trading stays off unless runtime mode is `live` **and** `TP_ENABLE_LIVE_TRADING=true`.
- Money values are stored as `Decimal` and persisted with SQLAlchemy `Numeric` columns.
- Configure `TP_ADMIN_PASSWORD_HASH` or set a non-default `TP_ADMIN_PASSWORD` before startup.
- Tests use SQLite + in-memory store/event bus fallback; Redis is optional during local development.
- The default SQLite database is single-writer and only suitable for local development; use PostgreSQL (`TP_DATABASE_URL`) for anything beyond that. Alembic migrations are the schema authority for production databases.
- In live mode (`TP_ENABLE_LIVE_TRADING=true`) the synthetic market data fallback is disabled: events fail (and are retried) instead of trading on synthetic prices.
- Old `event_outbox`, `pipeline_inbox`, `tradingview_events`, and `signals` rows are purged by the worker after `TP_RETENTION_DAYS` (default 30).
- Websocket broadcast subscribers only receive messages published after they connect; messages sent while disconnected are not replayed.
- The worker logs hot-spot metrics every `TP_METRICS_INTERVAL_SECONDS` (default 60): pipeline step timings, outbox lag, consumer-group pending count, and market-data cache hit rate.
