from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select

from app.core.module import AppContext, BaseModule, LoopSpec
from app.models import Position


class PositionService:
    async def reconcile(self, ctx: AppContext) -> None:
        async with ctx.session_factory() as session:
            result = await session.execute(select(Position).where(Position.status == "OPEN"))
            positions = list(result.scalars())
            if not positions:
                return
            prices = await self._latest_prices(ctx, {position.symbol for position in positions})
            changed = False
            for position in positions:
                price = prices.get(position.symbol)
                if price is None:
                    continue
                pnl = self._realized_pnl(
                    position.side,
                    position.entry_price,
                    price,
                    position.quantity,
                )
                if position.take_profit is not None and (
                    (position.side == "BUY" and price >= position.take_profit)
                    or (position.side == "SELL" and price <= position.take_profit)
                ):
                    position.status = "CLOSED"
                    position.realized_pnl = pnl
                    position.closed_at = datetime.now(UTC)
                    changed = True
                elif position.stop_loss is not None and (
                    (position.side == "BUY" and price <= position.stop_loss)
                    or (position.side == "SELL" and price >= position.stop_loss)
                ):
                    position.status = "CLOSED"
                    position.realized_pnl = pnl
                    position.closed_at = datetime.now(UTC)
                    changed = True
            if changed:
                await session.commit()

    @staticmethod
    async def _latest_prices(ctx: AppContext, symbols: set[str]) -> dict[str, Decimal]:
        """Resolve latest prices from the shared store, falling back to process state."""
        prices: dict[str, Decimal] = {}
        local_prices = ctx.state.get("latest_prices", {})
        store = ctx.state.get("store")
        for symbol in symbols:
            value = await store.get(f"price:{symbol}") if store is not None else None
            if value is not None:
                prices[symbol] = Decimal(value)
            elif symbol in local_prices:
                prices[symbol] = local_prices[symbol]
        return prices

    @staticmethod
    def _realized_pnl(
        side: str,
        entry_price: Decimal,
        exit_price: Decimal,
        quantity: Decimal,
    ) -> Decimal:
        if side == "SELL":
            return (entry_price - exit_price) * quantity
        return (exit_price - entry_price) * quantity


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
