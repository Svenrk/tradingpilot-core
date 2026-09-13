"""Train (and optionally backfill data for) the LightGBM strategy model.

Examples::

    # Pull 3000 hourly BTCUSDT bars from Binance into market_bars, then train.
    python -m workers.ml_train --symbol BTCUSDT --timeframe 1h --backfill 3000

    # Retrain from bars already persisted by the pipeline.
    python -m workers.ml_train --symbol ETHUSDT --timeframe 15m --horizon 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx

import app.models  # noqa: F401
from app.config import get_settings
from app.db import get_session_factory, init_db
from app.modules.market_data import Bar, BinanceMarketDataProvider, timeframe_to_ms
from app.modules.ml_strategy.labels import LabelConfig
from app.modules.ml_strategy.model import model_path
from app.modules.ml_strategy.store import count_bars, load_bar_arrays, upsert_bars
from app.modules.ml_strategy.training import InsufficientDataError, TrainingConfig, train_strategy
from app.observability import configure_logging

logger = logging.getLogger("ml_train")

BINANCE_MAX_LIMIT = 1000


async def backfill_bars(session_factory, symbol: str, timeframe: str, total: int) -> int:
    """Page backwards through Binance klines until ``total`` bars are stored."""
    provider = BinanceMarketDataProvider(httpx.AsyncClient(timeout=10.0))
    step_ms = timeframe_to_ms(timeframe)
    stored = 0
    end_time: int | None = None
    try:
        while stored < total:
            limit = min(BINANCE_MAX_LIMIT, total - stored)
            bars: list[Bar] = await provider.get_bars(symbol, timeframe, limit, end_time=end_time)
            if not bars:
                break
            async with session_factory() as session:
                stored += await upsert_bars(session, symbol, timeframe, bars)
                await session.commit()
            end_time = bars[0].open_time - step_ms
            logger.info("Backfilled %s/%s bars for %s/%s", stored, total, symbol, timeframe)
            if len(bars) < limit:
                break
    finally:
        await provider.close()
    return stored


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    symbol = args.symbol.upper()
    session_factory = get_session_factory(settings.database_url)
    await init_db(settings.database_url)

    if args.backfill:
        await backfill_bars(session_factory, symbol, args.timeframe, args.backfill)

    async with session_factory() as session:
        available = await count_bars(session, symbol, args.timeframe)
        arrays = await load_bar_arrays(session, symbol, args.timeframe, limit=args.max_bars)
    logger.info(
        "Loaded %s of %s persisted bars for %s/%s", len(arrays), available, symbol, args.timeframe
    )

    config = TrainingConfig(
        label=LabelConfig(
            horizon=args.horizon,
            take_profit_mult=args.take_profit_mult,
            stop_loss_mult=args.stop_loss_mult,
        ),
        n_folds=args.folds,
        cost_bps=args.cost_bps,
        min_train_size=args.min_train_size,
    )
    try:
        model, report = train_strategy(
            arrays, symbol=symbol, timeframe=args.timeframe, config=config
        )
    except InsufficientDataError as exc:
        logger.error("Cannot train: %s (try --backfill)", exc)
        return 2

    out_dir = Path(args.out or settings.ml_model_dir)
    path = model_path(out_dir, symbol, args.timeframe)
    if args.dry_run:
        logger.info("Dry run: model not written")
    else:
        model.save(path)
        logger.info("Saved model to %s", path)
    print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True, help="Binance interval, e.g. 1h")
    parser.add_argument("--backfill", type=int, default=0, help="Fetch N bars from Binance first")
    parser.add_argument(
        "--max-bars", type=int, default=None, help="Train on at most the latest N bars"
    )
    parser.add_argument("--horizon", type=int, default=12, help="Vertical barrier in bars")
    parser.add_argument(
        "--take-profit-mult", type=float, default=2.0, help="TP barrier in vol units"
    )
    parser.add_argument("--stop-loss-mult", type=float, default=1.0, help="SL barrier in vol units")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-size", type=int, default=300)
    parser.add_argument("--cost-bps", type=float, default=5.0, help="One-way cost in basis points")
    parser.add_argument("--out", default=None, help="Model directory (default TP_ML_MODEL_DIR)")
    parser.add_argument("--dry-run", action="store_true", help="Train and report without saving")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
