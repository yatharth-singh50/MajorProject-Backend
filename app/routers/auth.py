from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, get_current_user
from app.crud import users as users_crud
from app.db.mongodb import get_db
from app.db.postgres import get_session_dep
from app.models.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut
from app.models.sql_models import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, session: AsyncSession = Depends(get_session_dep)):
    if await users_crud.get_user_by_username(session, payload.username):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Username already taken")
    if await users_crud.get_user_by_email(session, payload.email):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Email already registered")

    user = await users_crud.create_user(
        session,
        username=payload.username,
        email=payload.email,
        password=payload.password,
        display_name=payload.display_name,
    )
    token = create_access_token(subject=user.username)
    return TokenResponse(access_token=token, user=UserOut(**user.to_public_dict(), trustScore=None))


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, session: AsyncSession = Depends(get_session_dep)):
    user = await users_crud.authenticate_user(session, payload.identifier, payload.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Incorrect username/email or password")
    token = create_access_token(subject=user.username)

    from app.crud.posts import trust_score_for  # local import avoids a circular import at module load

    trust_score = await trust_score_for(get_db(), user.id)
    return TokenResponse(access_token=token, user=UserOut(**user.to_public_dict(), trustScore=trust_score))


@router.get("/me", response_model=UserOut)
async def read_current_user(current_user: User = Depends(get_current_user)):
    from app.crud.posts import trust_score_for

    trust_score = await trust_score_for(get_db(), current_user.id)
    return UserOut(**current_user.to_public_dict(), trustScore=trust_score)
