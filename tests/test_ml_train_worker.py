from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import select

import workers.ml_train as ml_train
from app.config import get_settings
from app.db import get_session_factory, init_db
from app.models import MLTrainingRun
from app.modules.market_data import Bar
from app.modules.ml_strategy.store import upsert_bars
from app.modules.ml_strategy.training import train_strategy as real_train_strategy
from tests.test_ml_training import FAST_CONFIG, regime_arrays


def bars_from_arrays(arrays) -> list[Bar]:
    return [
        Bar(
            open_time=int(arrays.open_time[index]),
            open=Decimal(str(round(arrays.open[index], 8))),
            high=Decimal(str(round(arrays.high[index], 8))),
            low=Decimal(str(round(arrays.low[index], 8))),
            close=Decimal(str(round(arrays.close[index], 8))),
            volume=Decimal(str(round(arrays.volume[index], 8))),
        )
        for index in range(len(arrays))
    ]


async def test_ml_train_run_marks_failed_on_insufficient_data(
    test_env, test_db_url: str, tmp_path: Path
) -> None:
    settings = get_settings()
    settings.ml_model_dir = str(tmp_path / "models")
    await init_db(test_db_url)
    async with get_session_factory(test_db_url)() as session:
        await upsert_bars(session, "BTCUSDT", "1h", bars_from_arrays(regime_arrays(n=80)))
        await session.commit()

    args = ml_train.build_parser().parse_args(["--symbol", "BTCUSDT", "--timeframe", "1h", "--dry-run"])
    exit_code = await ml_train.run(args)
    assert exit_code == 2

    async with get_session_factory(test_db_url)() as session:
        run = await session.scalar(select(MLTrainingRun).order_by(MLTrainingRun.id.desc()))
    assert run is not None
    assert run.status == "failed"
    assert "Need at least" in run.error
    assert run.config["dry_run"] is True
    assert run.finished_at is not None


async def test_ml_train_run_marks_succeeded_and_preserves_stdout_json(
    test_env, test_db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    settings = get_settings()
    settings.ml_model_dir = str(tmp_path / "models")
    await init_db(test_db_url)
    async with get_session_factory(test_db_url)() as session:
        await upsert_bars(session, "BTCUSDT", "1h", bars_from_arrays(regime_arrays(n=900)))
        await session.commit()

    def fast_train_strategy(arrays, *, symbol: str, timeframe: str, config, on_progress=None):
        return real_train_strategy(
            arrays,
            symbol=symbol,
            timeframe=timeframe,
            config=FAST_CONFIG,
            on_progress=on_progress,
        )

    monkeypatch.setattr(ml_train, "train_strategy", fast_train_strategy)
    args = ml_train.build_parser().parse_args(["--symbol", "BTCUSDT", "--timeframe", "1h", "--dry-run"])
    exit_code = await ml_train.run(args)
    assert exit_code == 0

    report = json.loads(capsys.readouterr().out)
    assert "oos_accuracy" in report

    async with get_session_factory(test_db_url)() as session:
        run = await session.scalar(select(MLTrainingRun).order_by(MLTrainingRun.id.desc()))
    assert run is not None
    assert run.status == "succeeded"
    assert run.metrics is not None
    assert run.model_path is None
    assert run.config["dry_run"] is True
    assert run.n_samples == report["n_samples"]
