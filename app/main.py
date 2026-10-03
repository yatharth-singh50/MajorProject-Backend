"""
Application entrypoint. Run with:

    uvicorn app.main:app --reload

See README.md for full setup instructions (Mongo/Postgres, env vars, and
where to point the model checkpoint).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.db.mongodb import close_mongo, init_mongo
from app.db.postgres import close_postgres, init_postgres
from app.routers import analyze, auth, discovery, feed, notifications, posts, users, ws

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---------------------------------------------------------
    try:
        await init_mongo()
        logger.info("Connected to MongoDB")
    except Exception:  # noqa: BLE001
        logger.exception(
            "Could not connect to MongoDB at %s -- posts/feed endpoints will fail until it's reachable.",
            settings.MONGODB_URI,
        )

    try:
        await init_postgres()
        logger.info("Connected to Postgres and ensured tables exist")
    except Exception:  # noqa: BLE001
        logger.exception(
            "Could not connect to Postgres at %s -- auth/user endpoints will fail until it's reachable.",
            settings.POSTGRES_URI,
        )

    yield

    # --- Shutdown ----------------------------------------------------------
    await close_mongo()
    await close_postgres()


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Backend for the Sāthi fake-news-detection social app: auth, profiles, "
        "posts/replies, and a MuRIL-based multilingual fact-checking pipeline."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(posts.router)
app.include_router(feed.router)
app.include_router(discovery.router)
app.include_router(analyze.router)
app.include_router(notifications.router)
app.include_router(ws.router)


@app.get("/health", tags=["health"])
async def health_check():
    from app.services.text_classifier import classifier_backend_name

    return {
        "status": "ok",
        "text_classifier_backend": classifier_backend_name(),
    }
