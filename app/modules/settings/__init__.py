from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict

from app.core.module import AppContext, BaseModule
from app.security import require_role

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    mode: Literal["paper", "live"] = "paper"
    kill_switch: bool = False
    max_positions: int = 5
    daily_loss_limit: Decimal = Decimal("1000")
    min_reward_risk: Decimal = Decimal("1.5")
    exposure_cap: Decimal = Decimal("100000")


class SettingsService:
    def __init__(self, store, app_settings) -> None:
        self.store = store
        self.app_settings = app_settings
        self.key = "settings:runtime"

    async def get_runtime_settings(self) -> RuntimeSettings:
        payload = await self.store.get(self.key)
        return RuntimeSettings.model_validate_json(payload) if payload else RuntimeSettings()

    async def update_runtime_settings(self, payload: RuntimeSettings) -> RuntimeSettings:
        await self.store.set(self.key, payload.model_dump_json())
        return payload

    async def is_live(self) -> bool:
        runtime = await self.get_runtime_settings()
        return runtime.mode == "live" and self.app_settings.enable_live_trading


def _service(request: Request) -> SettingsService:
    return request.app.state.ctx.state["settings_service"]


@router.get("", response_model=RuntimeSettings)
async def get_runtime_settings(
    service: Annotated[SettingsService, Depends(_service)],
    session=Depends(require_role("read", "paper", "live")),
) -> RuntimeSettings:
    del session
    return await service.get_runtime_settings()


@router.put("", response_model=RuntimeSettings)
async def update_runtime_settings(
    payload: RuntimeSettings,
    service: Annotated[SettingsService, Depends(_service)],
    session=Depends(require_role("paper", "live")),
) -> RuntimeSettings:
    del session
    return await service.update_runtime_settings(payload)


class SettingsModule(BaseModule):
    name = "settings"

    async def on_startup(self, ctx: AppContext) -> None:
        ctx.state["settings_service"] = SettingsService(ctx.state["store"], ctx.settings)

    def routers(self):
        return [router]


module = SettingsModule()
