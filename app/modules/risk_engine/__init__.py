from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import func, select

from app.core.module import BaseModule, PipelineContext, PipelineStep
from app.models import Position


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    reasons: list[str]


class RiskService:
    async def evaluate(self, ctx: PipelineContext) -> RiskDecision:
        signal = ctx.signal
        if signal is None or signal.action == "HOLD":
            return RiskDecision(False, ["No executable signal"])
        settings_service = ctx.app.state["settings_service"]
        runtime = await settings_service.get_runtime_settings()
        reasons: list[str] = []
        if runtime.kill_switch:
            reasons.append("Kill switch enabled")
        if signal.stop_loss is None or signal.take_profit is None:
            reasons.append("Missing stop-loss or take-profit")
        else:
            entry = ctx.snapshot.price
            risk = abs(entry - signal.stop_loss)
            reward = abs(signal.take_profit - entry)
            if risk <= 0:
                reasons.append("Invalid stop-loss")
            elif reward / risk < runtime.min_reward_risk:
                reasons.append("Reward-to-risk below minimum")
        open_positions = await ctx.session.scalar(
            select(func.count()).select_from(Position).where(Position.status == "OPEN")
        )
        if open_positions and open_positions >= runtime.max_positions:
            reasons.append("Max positions reached")
        day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        daily_loss = await ctx.session.scalar(
            select(func.coalesce(func.sum(Position.realized_pnl), 0)).where(
                Position.status == "CLOSED",
                Position.closed_at.is_not(None),
                Position.closed_at >= day_start,
            )
        )
        if Decimal(daily_loss or 0) <= (Decimal("0") - runtime.daily_loss_limit):
            reasons.append("Daily loss limit breached")
        exposure = await ctx.session.scalar(
            select(func.coalesce(func.sum(Position.entry_price * Position.quantity), 0)).where(
                Position.status == "OPEN"
            )
        )
        projected = Decimal(exposure or 0) + (ctx.snapshot.price * Decimal("1"))
        if projected > runtime.exposure_cap:
            reasons.append("Exposure cap exceeded")
        return RiskDecision(not reasons, reasons or ["Approved"])


async def evaluate_risk_step(ctx: PipelineContext) -> None:
    service = ctx.app.state["risk_service"]
    ctx.risk_decision = await service.evaluate(ctx)


class RiskEngineModule(BaseModule):
    name = "risk_engine"
    depends_on = ("settings", "signal_engine")

    async def on_startup(self, ctx) -> None:
        ctx.state["risk_service"] = RiskService()

    def pipeline_steps(self):
        return [PipelineStep(name="risk_gate", order=30, handler=evaluate_risk_step)]


module = RiskEngineModule()
