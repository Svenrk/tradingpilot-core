# TradingPilot Core

TradingPilot Core is a stripped, production-shaped FastAPI backend built around plugin-style modules. Every feature lives in `app/modules/<name>/` and is enabled or disabled through `TP_ENABLED_MODULES`.

## Architecture

```text
TradingView -> /api/v1/tv/webhook/{webhook_id}
           -> PostgreSQL tradingview_events + event_outbox
           -> Redis Stream tv_events
           -> pipeline worker (market_data -> signal_engine -> ml_strategy -> risk_engine -> execution)
           -> PostgreSQL signals/orders/positions/market_bars
           -> Redis Stream ui_updates
           -> /api/v1/events|signals|orders|positions|ml + /api/v1/ws/updates
```

## ML strategy (`ml_strategy` module)

A LightGBM strategy that runs beside the rule-based `signal_engine` and learns BUY/SELL/HOLD from persisted OHLCV bars.

**How it works**

- `bars_persist` (pipeline order 15) upserts every snapshot's OHLCV candles into `market_bars`, so the training corpus grows as webhooks arrive.
- 42 strictly causal features (multi-lag returns, realised-vol ratios, EMA/MACD/RSI/Bollinger/ATR/Donchian geometry, candle anatomy, volume z-scores, return skew/kurtosis/autocorrelation, variance ratio, time-of-day/week) are computed with pure numpy so training and inference share one code path.
- Labels use the **triple-barrier method**: a volatility-scaled take-profit (2σ), stop-loss (1σ) and a time barrier (`--horizon` bars). The same barriers become the live signal's `take_profit` / `stop_loss`, so the risk gate sees exactly what the model was trained to predict.
- Training is **purged, embargoed walk-forward CV** (no label overlap between folds), with class/return-magnitude sample weights and early stopping. Trade thresholds (`min_probability`, `min_edge`) are selected only on out-of-sample forecasts by maximising Sharpe after costs, then a final model is refit on all data.
- Models are stored as `models/<SYMBOL>_<timeframe>.json` (booster + metadata + OOS metrics) and hot-reloaded when the file changes.
- `ml_signal` (order 25) writes its decision as a `Signal` row and combines it with the rule signal according to `TP_ML_STRATEGY_MODE`:
  - `shadow` (default) — record only; the rule-based signal still drives execution.
  - `primary` — the ML decision replaces the rule-based signal.
  - `confirm` — trade only when both agree; otherwise HOLD.

**Train a model**

```bash
# Bootstrap 3000 hourly bars from Binance into market_bars and train
docker compose run --rm worker python -m workers.ml_train --symbol BTCUSDT --timeframe 1h --backfill 3000

# Retrain later from bars the pipeline has persisted
docker compose run --rm worker python -m workers.ml_train --symbol BTCUSDT --timeframe 1h
```

The command prints a JSON report (fold metrics, OOS log-loss/accuracy, selected thresholds, OOS backtest vs. an unthresholded baseline, feature importance). Use `--dry-run` to evaluate without saving. Both `api` and `worker` mount `./models`, so a newly written model is picked up without a restart.

**Inspect**

- `GET /api/v1/ml/models` — loaded models with thresholds and OOS metrics.
- `GET /api/v1/ml/predict/{symbol}/{timeframe}` — current probabilities and decision from persisted bars.
- `POST /api/v1/ml/models/reload` — force a reload.

Settings: `TP_ML_STRATEGY_MODE`, `TP_ML_MODEL_DIR`, `TP_ML_INFERENCE_BARS` (bars loaded for inference, must exceed the 64-bar warm-up), `TP_ML_PERSIST_BARS`, `TP_MARKET_DATA_BAR_LIMIT`.

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
