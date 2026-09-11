from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from app.config import get_settings
from app.core.module import AppContext, BaseModule
from app.security import (
    SessionData,
    create_session,
    delete_session,
    hash_password,
    require_session,
    verify_password,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    user_id: str
    role: str
    csrf_token: str


@router.post("/login", response_model=UserResponse)
async def login(payload: LoginRequest, request: Request, response: Response) -> UserResponse:
    ctx = request.app.state.ctx
    settings = get_settings()
    password_hash = ctx.state["auth_password_hash"]
    if payload.username != settings.admin_username or not verify_password(
        payload.password, password_hash
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    session = SessionData(
        user_id=payload.username,
        role=settings.admin_role,
        csrf_token=ctx.state["csrf_token_factory"](),
    )
    session_id = await create_session(ctx.state["store"], settings, session)
    response.set_cookie(
        settings.session_cookie_name,
        session_id,
        httponly=True,
        samesite="strict",
        secure=True,
    )
    return UserResponse.model_validate(session.model_dump())


@router.get("/me", response_model=UserResponse)
async def me(session: Annotated[SessionData, Depends(require_session)]) -> UserResponse:
    return UserResponse.model_validate(session.model_dump())


@router.post("/logout")
async def logout(
    request: Request, response: Response, session: Annotated[SessionData, Depends(require_session)]
) -> dict[str, str]:
    del session
    settings = get_settings()
    session_id = request.cookies.get(settings.session_cookie_name)
    await delete_session(request.app.state.ctx.state["store"], session_id)
    response.delete_cookie(settings.session_cookie_name)
    return {"status": "logged_out"}


class AuthModule(BaseModule):
    name = "auth"

    async def on_startup(self, ctx: AppContext) -> None:
        settings = get_settings()
        if settings.admin_password_hash:
            ctx.state["auth_password_hash"] = settings.admin_password_hash
        elif settings.admin_password not in {"admin", "change-me-now"}:
            ctx.state["auth_password_hash"] = hash_password(settings.admin_password)
        else:
            raise RuntimeError(
                "Configure TP_ADMIN_PASSWORD_HASH or a non-default TP_ADMIN_PASSWORD before startup"
            )
        ctx.state["csrf_token_factory"] = __import__("secrets").token_urlsafe

    def routers(self):
        return [router]


module = AuthModule()
