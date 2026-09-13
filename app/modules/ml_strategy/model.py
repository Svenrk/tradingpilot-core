"""Serialisable LightGBM strategy model and its trade-decision logic."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from app.modules.ml_strategy.backtest import DecisionThresholds, positions_from_probabilities
from app.modules.ml_strategy.features import (
    FEATURE_NAMES,
    WARMUP_BARS,
    BarArrays,
    compute_features,
    realized_volatility,
)
from app.modules.ml_strategy.labels import CLASS_DOWN, CLASS_FLAT, CLASS_UP, LabelConfig
from app.modules.signal_engine import SignalDecision
from app.money import quantize

logger = logging.getLogger(__name__)

MODEL_FORMAT_VERSION = 1


def _require_lightgbm():
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover - exercised only without the dependency
        raise RuntimeError(
            "lightgbm is required for the ml_strategy module; install requirements.txt"
        ) from exc
    return lightgbm


@dataclass(slots=True)
class ModelMetadata:
    symbol: str
    timeframe: str
    feature_names: list[str]
    label_config: LabelConfig
    thresholds: DecisionThresholds
    trained_at: str
    n_samples: int
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    format_version: int = MODEL_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "feature_names": self.feature_names,
            "label_config": {
                "horizon": self.label_config.horizon,
                "take_profit_mult": self.label_config.take_profit_mult,
                "stop_loss_mult": self.label_config.stop_loss_mult,
                "vol_window": self.label_config.vol_window,
                "min_vol": self.label_config.min_vol,
            },
            "thresholds": self.thresholds.to_dict(),
            "trained_at": self.trained_at,
            "n_samples": self.n_samples,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "format_version": self.format_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        return cls(
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            feature_names=list(data["feature_names"]),
            label_config=LabelConfig(**data["label_config"]),
            thresholds=DecisionThresholds(**data["thresholds"]),
            trained_at=data["trained_at"],
            n_samples=int(data["n_samples"]),
            metrics=dict(data.get("metrics", {})),
            feature_importance=dict(data.get("feature_importance", {})),
            format_version=int(data.get("format_version", MODEL_FORMAT_VERSION)),
        )


@dataclass(slots=True)
class Prediction:
    probabilities: np.ndarray
    position: int
    volatility: float

    @property
    def p_down(self) -> float:
        return float(self.probabilities[CLASS_DOWN])

    @property
    def p_flat(self) -> float:
        return float(self.probabilities[CLASS_FLAT])

    @property
    def p_up(self) -> float:
        return float(self.probabilities[CLASS_UP])


class StrategyModel:
    """A trained booster plus everything needed to reproduce its decisions."""

    def __init__(self, booster: Any, metadata: ModelMetadata) -> None:
        self.booster = booster
        self.metadata = metadata

    @property
    def key(self) -> str:
        return model_key(self.metadata.symbol, self.metadata.timeframe)

    @property
    def min_bars(self) -> int:
        return WARMUP_BARS + 1

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(features)
        return np.asarray(self.booster.predict(matrix, num_threads=1), dtype=float)

    def predict_latest(self, arrays: BarArrays) -> Prediction | None:
        if len(arrays) < self.min_bars:
            return None
        matrix, names = compute_features(arrays)
        if names != self.metadata.feature_names:
            raise ValueError("Feature set drifted from the trained model; retrain the model")
        latest = matrix[-1]
        if not np.isfinite(latest).any():
            return None
        probabilities = self.predict_proba(latest)[0]
        position = int(
            positions_from_probabilities(probabilities[None, :], self.metadata.thresholds)[0]
        )
        vol = realized_volatility(arrays.close, self.metadata.label_config.vol_window)[-1]
        return Prediction(probabilities=probabilities, position=position, volatility=float(vol))

    def decide(self, prediction: Prediction | None, price: Decimal) -> SignalDecision:
        if prediction is None:
            return SignalDecision("HOLD", "ml: insufficient bar history", Decimal("0"))
        confidence = quantize(max(prediction.p_up, prediction.p_down))
        rationale = (
            f"ml: p_up={prediction.p_up:.3f} p_down={prediction.p_down:.3f} "
            f"p_flat={prediction.p_flat:.3f}"
        )
        if prediction.position == 0:
            return SignalDecision("HOLD", rationale + " below threshold", confidence)
        tp_width, sl_width = self.metadata.label_config.barrier_widths(prediction.volatility)
        price_f = float(price)
        if prediction.position > 0:
            stop_loss = quantize(price_f * float(np.exp(-sl_width)))
            take_profit = quantize(price_f * float(np.exp(tp_width)))
            return SignalDecision("BUY", rationale, confidence, stop_loss, take_profit)
        stop_loss = quantize(price_f * float(np.exp(sl_width)))
        take_profit = quantize(price_f * float(np.exp(-tp_width)))
        return SignalDecision("SELL", rationale, confidence, stop_loss, take_profit)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metadata": self.metadata.to_dict(), "booster": self.booster.model_to_string()}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> StrategyModel:
        lightgbm = _require_lightgbm()
        payload = json.loads(Path(path).read_text())
        metadata = ModelMetadata.from_dict(payload["metadata"])
        if metadata.format_version != MODEL_FORMAT_VERSION:
            raise ValueError(f"Unsupported model format version {metadata.format_version}")
        if metadata.feature_names != FEATURE_NAMES:
            raise ValueError(f"Model {path} was trained on a different feature set; retrain")
        booster = lightgbm.Booster(model_str=payload["booster"])
        return cls(booster, metadata)


def model_key(symbol: str, timeframe: str) -> str:
    return f"{symbol.upper()}_{timeframe}"


def model_path(model_dir: Path, symbol: str, timeframe: str) -> Path:
    return Path(model_dir) / f"{model_key(symbol, timeframe)}.json"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
