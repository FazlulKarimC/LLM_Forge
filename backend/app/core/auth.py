"""Verify Clerk session JWTs without trusting browser-provided identity headers."""

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Identity:
    subject: str
    display_name: str


@lru_cache(maxsize=4)
def _jwks_client(issuer: str):
    return jwt.PyJWKClient(f"{issuer}/.well-known/jwks.json", cache_keys=True, timeout=5)


def verify_session_token(token: str) -> Identity:
    issuer = settings.CLERK_ISSUER_URL.rstrip("/")
    parsed = urlparse(issuer)
    if parsed.scheme != "https" or not parsed.hostname or parsed.query or parsed.fragment:
        raise HTTPException(503, "Authentication is not configured. Set CLERK_ISSUER_URL.")
    parties = [value.strip() for value in settings.CLERK_AUTHORIZED_PARTIES.split(",") if value.strip()]
    if not parties:
        raise HTTPException(503, "Authentication authorized parties are not configured.")
    try:
        key = settings.CLERK_JWT_PUBLIC_KEY.replace("\\n", "\n") or _jwks_client(issuer).get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token, key, algorithms=["RS256"], issuer=issuer,
            audience=settings.CLERK_AUDIENCE or None,
            options={"require": ["exp", "iat", "nbf", "iss", "sub"], "verify_aud": bool(settings.CLERK_AUDIENCE)},
            leeway=5,
        )
        if claims.get("azp") not in parties or not isinstance(claims.get("sub"), str) or not claims["sub"].startswith("user_"):
            raise jwt.InvalidTokenError("Invalid session identity or authorized party")
        if not isinstance(claims.get("sid"), str) or not claims["sid"]:
            raise jwt.InvalidTokenError("A session token is required")
    except jwt.PyJWKClientConnectionError as exc:
        raise HTTPException(503, "Authentication service unavailable. Try again.") from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(401, "Invalid or expired session", headers={"WWW-Authenticate": "Bearer"}) from exc
    name = claims.get("name")
    return Identity(claims["sub"], name[:120] if isinstance(name, str) and name.strip() else "Developer")


async def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Identity:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Sign in to access this workspace", headers={"WWW-Authenticate": "Bearer"})
    return await asyncio.to_thread(verify_session_token, credentials.credentials)
