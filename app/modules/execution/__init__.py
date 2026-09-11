from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.module import AppContext, BaseModule, PipelineContext, PipelineStep
from app.models import Order, Position


@dataclass(slots=True)
class BrokerOrderResult:
    order_id: str
    status: str
    filled_price: Decimal
    filled_quantity: Decimal


class Broker:
    async def place_order(
        self, order: Order
    ) -> BrokerOrderResult:  # pragma: no cover - interface-like
        raise NotImplementedError


class PaperBroker(Broker):
    async def place_order(self, order: Order) -> BrokerOrderResult:
        return BrokerOrderResult(
            order_id=order.client_order_id,
            status="filled",
            filled_price=order.price,
            filled_quantity=order.quantity,
        )


def create_broker(app_settings) -> Broker:
    del app_settings
    return PaperBroker()


class ExecutionService:
    def __init__(self, broker: Broker) -> None:
        self.broker = broker

    async def execute(self, ctx: PipelineContext) -> tuple[Order, Position] | None:
        if ctx.risk_decision is None or not ctx.risk_decision.approved:
            raise RuntimeError("Execution requires approved risk decision")
        if ctx.signal is None or ctx.signal.action == "HOLD":
            return None
        quantity = ctx.metadata.get("quantity", Decimal("1"))
        order = Order(
            client_order_id=str(uuid4()),
            symbol=ctx.snapshot.symbol,
            side=ctx.signal.action,
            status="pending",
            quantity=quantity,
            price=ctx.snapshot.price,
            stop_loss=ctx.signal.stop_loss,
            take_profit=ctx.signal.take_profit,
            broker="paper",
        )
        ctx.session.add(order)
        await ctx.session.flush()
        result = await self.broker.place_order(order)
        order.status = result.status.upper()
        order.price = result.filled_price
        position = await self._upsert_position(ctx, order, result)
        return order, position

    async def _upsert_position(
        self,
        ctx: PipelineContext,
        order: Order,
        result: BrokerOrderResult,
    ) -> Position:
        position = await ctx.session.scalar(
            select(Position)
            .where(Position.symbol == order.symbol, Position.status == "OPEN")
            .with_for_update()
        )
        if position is None:
            try:
                async with ctx.session.begin_nested():
                    position = Position(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=result.filled_quantity,
                        entry_price=result.filled_price,
                        stop_loss=order.stop_loss,
                        take_profit=order.take_profit,
                        status="OPEN",
                    )
                    ctx.session.add(position)
                    await ctx.session.flush()
                    return position
            except IntegrityError:
                position = await ctx.session.scalar(
                    select(Position)
                    .where(Position.symbol == order.symbol, Position.status == "OPEN")
                    .with_for_update()
                )
        if position is None:
            raise RuntimeError("Unable to load position after insert attempt")
        if position.side != order.side:
            raise RuntimeError("Conflicting open position exists for symbol")
        combined_quantity = position.quantity + result.filled_quantity
        position.entry_price = (
            (position.entry_price * position.quantity)
            + (result.filled_price * result.filled_quantity)
        ) / combined_quantity
        position.quantity = combined_quantity
        position.stop_loss = order.stop_loss
        position.take_profit = order.take_profit
        position.status = "OPEN"
        await ctx.session.flush()
        return position


async def execute_order_step(ctx: PipelineContext) -> None:
    service: ExecutionService = ctx.app.state["execution_service"]
    result = await service.execute(ctx)
    if result is not None:
        ctx.order, ctx.position = result


class ExecutionModule(BaseModule):
    name = "execution"
    depends_on = ("risk_engine",)

    async def on_startup(self, ctx: AppContext) -> None:
        broker = create_broker(ctx.settings)
        ctx.state["broker"] = broker
        ctx.state["execution_service"] = ExecutionService(broker)

    def models(self):
        return [Order, Position]

    def pipeline_steps(self):
        return [PipelineStep(name="execute_order", order=40, handler=execute_order_step)]


module = ExecutionModule()
