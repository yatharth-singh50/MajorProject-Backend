"""
Password hashing and JWT issuing/verification.

Kept deliberately small and dependency-light: passlib[bcrypt] for hashing,
python-jose for JWTs. `get_current_user` is the dependency every protected
route uses to resolve the caller from the `Authorization: Bearer <token>`
header.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from app.core.config import get_settings
from app.core.hashing import hash_password, verify_password  # re-exported for convenience
from app.models.sql_models import User
from app.crud import users as users_crud
from app.db.postgres import get_session_dep
from sqlalchemy.ext.asyncio import AsyncSession

settings = get_settings()

# tokenUrl is only used for OpenAPI docs' "Authorize" button; actual login
# happens via POST /auth/login which returns a JSON body, not a redirect.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

__all__ = [
    "require_admin",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "get_current_user",
    "get_current_user_optional",
]


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode = {"sub": subject, "exp": expire}
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_session_dep),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if token is None:
        raise credentials_exception

    username = decode_access_token(token)
    if username is None:
        raise credentials_exception

    user = await users_crud.get_user_by_username(session, username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_session_dep),
) -> Optional[User]:
    """Same as get_current_user but returns None instead of raising.

    Useful for endpoints that behave differently for logged-in users
    (e.g. `likedByMe`) but are still viewable while logged out.
    """
    if token is None:
        return None
    username = decode_access_token(token)
    if username is None:
        return None
    return await users_crud.get_user_by_username(session, username)


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Dependency for admin-only routes (see settings.ADMIN_USERNAMES)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admins only")
    return current_user
