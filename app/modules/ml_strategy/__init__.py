"""LightGBM strategy module: learns BUY/SELL/HOLD from persisted OHLCV bars.

Pipeline placement: ``bars_persist`` (15) runs after the market snapshot and
before the rule-based ``signal_evaluate`` (20); ``ml_signal`` (25) runs after
it and before the ``risk_gate`` (30). ``TP_ML_STRATEGY_MODE`` controls whether
the ML decision merely shadows, replaces, or must confirm the rule signal.

Train models with ``python -m workers.ml_train``.
"""

from __future__ import annotations

import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.core.module import AppContext, BaseModule, PipelineContext, PipelineStep
from app.models import MarketBar, Signal
from app.modules.ml_strategy.model import StrategyModel, model_key, model_path
from app.modules.ml_strategy.store import load_bar_arrays, upsert_bars
from app.modules.signal_engine import SignalDecision
from app.observability import metrics
from app.security import require_role

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])


class ModelRegistry:
    """Loads ``<SYMBOL>_<timeframe>.json`` models and hot-reloads on file change."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._models: dict[str, StrategyModel] = {}
        self._mtimes: dict[str, float] = {}

    def get(self, symbol: str, timeframe: str) -> StrategyModel | None:
        key = model_key(symbol, timeframe)
        path = model_path(self.model_dir, symbol, timeframe)
        try:
            mtime = os.stat(path).st_mtime
        except FileNotFoundError:
            self._models.pop(key, None)
            self._mtimes.pop(key, None)
            return None
        if self._mtimes.get(key) != mtime:
            try:
                self._models[key] = StrategyModel.load(path)
                self._mtimes[key] = mtime
                logger.info("Loaded ML model %s from %s", key, path)
            except Exception:
                logger.exception("Failed to load ML model %s", path)
                self._models.pop(key, None)
                self._mtimes[key] = mtime
                return None
        return self._models.get(key)

    def load_all(self) -> list[str]:
        if not self.model_dir.exists():
            return []
        loaded = []
        for path in sorted(self.model_dir.glob("*.json")):
            symbol, _, timeframe = path.stem.rpartition("_")
            if symbol and self.get(symbol, timeframe) is not None:
                loaded.append(model_key(symbol, timeframe))
        return loaded

    def reset(self) -> None:
        self._models.clear()
        self._mtimes.clear()

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "key": key,
                "symbol": model.metadata.symbol,
                "timeframe": model.metadata.timeframe,
                "trained_at": model.metadata.trained_at,
                "n_samples": model.metadata.n_samples,
                "thresholds": model.metadata.thresholds.to_dict(),
                "metrics": model.metadata.metrics,
            }
            for key, model in sorted(self._models.items())
        ]


def combine_decisions(mode: str, rule: SignalDecision | None, ml: SignalDecision) -> SignalDecision:
    """Resolve the effective signal from the rule-based and ML decisions."""
    if mode == "primary" or rule is None:
        return ml
    if mode == "confirm":
        if rule.action == ml.action and ml.action != "HOLD":
            strength = max(rule.strength, ml.strength)
            return SignalDecision(
                ml.action,
                f"{rule.rationale} | confirmed by {ml.rationale}",
                strength,
                ml.stop_loss,
                ml.take_profit,
            )
        return SignalDecision(
            "HOLD", f"rule={rule.action} ml={ml.action} disagree", Decimal("0.20")
        )
    return rule


async def persist_bars_step(ctx: PipelineContext) -> None:
    snapshot = ctx.snapshot
    if snapshot is None or not ctx.app.settings.ml_persist_bars:
        return
    # Flat bars synthesised from closes-only providers carry no OHLCV information.
    bars = [bar for bar in snapshot.bars if not bar.synthetic]
    if not bars:
        return
    written = await upsert_bars(ctx.session, snapshot.symbol, snapshot.timeframe, bars)
    metrics.increment("ml.bars_persisted", written)


async def ml_signal_step(ctx: PipelineContext) -> None:
    registry: ModelRegistry = ctx.app.state["ml_model_registry"]
    snapshot = ctx.snapshot
    if snapshot is None:
        return
    model = registry.get(snapshot.symbol, snapshot.timeframe)
    if model is None:
        metrics.increment("ml.no_model")
        return
    arrays = await load_bar_arrays(
        ctx.session,
        snapshot.symbol,
        snapshot.timeframe,
        limit=ctx.app.settings.ml_inference_bars,
    )
    prediction = model.predict_latest(arrays)
    decision = model.decide(prediction, snapshot.price)
    metrics.increment(f"ml.decision.{decision.action.lower()}")
    ctx.metadata["ml_decision"] = decision
    if prediction is not None:
        ctx.metadata["ml_probabilities"] = {
            "down": prediction.p_down,
            "flat": prediction.p_flat,
            "up": prediction.p_up,
        }
    mode = ctx.app.settings.ml_strategy_mode
    ctx.signal = combine_decisions(mode, ctx.signal, decision)
    ctx.session.add(
        Signal(
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            action=decision.action,
            strength=decision.strength,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            rationale=f"[{mode}] {decision.rationale}",
        )
    )
    await ctx.session.flush()


class ModelSummary(BaseModel):
    key: str
    symbol: str
    timeframe: str
    trained_at: str
    n_samples: int
    thresholds: dict[str, float]
    metrics: dict[str, Any]


class PredictionResponse(BaseModel):
    symbol: str
    timeframe: str
    action: str
    strength: Decimal
    stop_loss: Decimal | None
    take_profit: Decimal | None
    rationale: str
    probabilities: dict[str, float] | None
    bars_used: int


def _registry(request: Request) -> ModelRegistry:
    return request.app.state.ctx.state["ml_model_registry"]


@router.get("/models", response_model=list[ModelSummary])
async def list_models(
    registry: Annotated[ModelRegistry, Depends(_registry)],
    session=Depends(require_role("read", "paper", "live")),
) -> list[dict[str, Any]]:
    del session
    registry.load_all()
    return registry.describe()


@router.post("/models/reload", response_model=list[ModelSummary])
async def reload_models(
    registry: Annotated[ModelRegistry, Depends(_registry)],
    session=Depends(require_role("paper", "live")),
) -> list[dict[str, Any]]:
    del session
    registry.reset()
    registry.load_all()
    return registry.describe()


@router.get("/predict/{symbol}/{timeframe}", response_model=PredictionResponse)
async def predict(
    symbol: str,
    timeframe: str,
    request: Request,
    registry: Annotated[ModelRegistry, Depends(_registry)],
    session=Depends(require_role("read", "paper", "live")),
) -> dict[str, Any]:
    del session
    ctx: AppContext = request.app.state.ctx
    model = registry.get(symbol, timeframe)
    if model is None:
        raise HTTPException(status_code=404, detail="No trained model for this market")
    async with ctx.session_factory() as db:
        arrays = await load_bar_arrays(
            db, symbol.upper(), timeframe, limit=ctx.settings.ml_inference_bars
        )
    if len(arrays) == 0:
        raise HTTPException(status_code=404, detail="No persisted bars for this market")
    prediction = model.predict_latest(arrays)
    decision = model.decide(prediction, Decimal(str(arrays.close[-1])))
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "action": decision.action,
        "strength": decision.strength,
        "stop_loss": decision.stop_loss,
        "take_profit": decision.take_profit,
        "rationale": decision.rationale,
        "probabilities": None
        if prediction is None
        else {"down": prediction.p_down, "flat": prediction.p_flat, "up": prediction.p_up},
        "bars_used": len(arrays),
    }


class MLStrategyModule(BaseModule):
    name = "ml_strategy"
    depends_on = ("market_data", "signal_engine")

    async def on_startup(self, ctx: AppContext) -> None:
        registry = ModelRegistry(Path(ctx.settings.ml_model_dir))
        loaded = registry.load_all()
        logger.info(
            "ML strategy mode=%s models=%s", ctx.settings.ml_strategy_mode, loaded or "none"
        )
        ctx.state["ml_model_registry"] = registry

    def models(self):
        return [MarketBar]

    def routers(self):
        return [router]

    def pipeline_steps(self):
        return [
            PipelineStep(name="bars_persist", order=15, handler=persist_bars_step),
            PipelineStep(name="ml_signal", order=25, handler=ml_signal_step),
        ]


module = MLStrategyModule()
