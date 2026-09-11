from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select

from app.core.module import AppContext, BaseModule, LoopSpec
from app.models import Position


class PositionService:
    async def reconcile(self, ctx: AppContext) -> None:
        latest_prices = ctx.state.get("latest_prices", {})
        async with ctx.session_factory() as session:
            result = await session.execute(select(Position).where(Position.status == "OPEN"))
            for position in result.scalars():
                price = latest_prices.get(position.symbol)
                if price is None:
                    continue
                if position.take_profit is not None and (
                    (position.side == "BUY" and price >= position.take_profit)
                    or (position.side == "SELL" and price <= position.take_profit)
                ):
                    position.status = "CLOSED"
                    position.realized_pnl = abs((price - position.entry_price) * position.quantity)
                    position.closed_at = datetime.now(UTC)
                elif position.stop_loss is not None and (
                    (position.side == "BUY" and price <= position.stop_loss)
                    or (position.side == "SELL" and price >= position.stop_loss)
                ):
                    position.status = "CLOSED"
                    position.realized_pnl = Decimal("0") - abs(
                        (price - position.entry_price) * position.quantity
                    )
                    position.closed_at = datetime.now(UTC)
            await session.commit()


async def reconcile_positions(ctx: AppContext) -> None:
    service: PositionService = ctx.state["position_service"]
    await service.reconcile(ctx)


class PositionsModule(BaseModule):
    name = "positions"
    depends_on = ("execution",)

    async def on_startup(self, ctx: AppContext) -> None:
        ctx.state["position_service"] = PositionService()

    def background_loops(self):
        return [LoopSpec(name="positions", coro_factory=reconcile_positions, interval_seconds=5.0)]


module = PositionsModule()
