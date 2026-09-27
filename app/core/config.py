"""
Central application configuration.

Everything here is overridable via environment variables or a `.env` file
(see `.env.example` at the project root). Nothing sensitive is hardcoded.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- General -------------------------------------------------------
    APP_NAME: str = "Sathi Backend"
    ENV: str = Field(default="development")  # development | staging | production
    DEBUG: bool = True
    API_PREFIX: str = ""

    # --- CORS ------------------------------------------------------------
    # The Vite dev server defaults to 5173; add your deployed frontend origin(s)
    # in production via the CORS_ORIGINS env var (comma-separated).
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # --- Mongo (posts, comments, notifications, media) --------------------
    # MongoDB is used for the high-write, flexible-shape social graph data:
    # posts, replies/comments, likes/reposts, notifications, and the ML
    # analysis payload attached to each post.
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "sathi"

    # --- Postgres / Supabase (users, auth) --------------------------------
    # Relational Postgres (optionally hosted on Supabase) is used for
    # accounts/auth: usernames must be unique, passwords hashed, and this
    # data benefits from strong relational constraints. Supply either a full
    # Supabase connection string, or point at a local Postgres for dev.
    #
    # Example Supabase URI:
    #   postgresql+asyncpg://postgres:<password>@<project>.supabase.co:5432/postgres
    POSTGRES_URI: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/sathi"

    # --- Auth / JWT --------------------------------------------------------
    JWT_SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # --- Media / uploads -----------------------------------------------
    # Images are accepted as base64-encoded byte strings and stored inline
    # on the post document (fine for a prototype / moderate media sizes).
    # For heavier production use, swap `utils/media.py` for a call to
    # object storage (S3, Supabase Storage, etc.) and store a URL instead.
    MAX_IMAGE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB
    ALLOWED_IMAGE_MIME_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp", "image/gif"]

    # --- ML: text classifier (MuRIL) --------------------------------------
    # Points at the checkpoint produced by the model-training repo
    # (see major-project-model-training/src/train.py -> OUTPUT_DIR).
    # If this path doesn't exist, or torch/transformers aren't installed,
    # the service transparently falls back to a lightweight heuristic
    # classifier so the API keeps functioning end-to-end.
    TEXT_MODEL_PATH: str = "./models/muril_combined"
    TEXT_MODEL_MAX_LENGTH: int = 192
    TEXT_MODEL_NAME: str = "MuRIL-FND v1"

    # --- ML: image encoder (ViT) -----------------------------------------
    # Feature-extraction only for now (see the training repo's
    # demo_multimodal.py) -- multimodal fusion is a future stage, not yet
    # trained. Kept behind a flag so it can be disabled cheaply.
    ENABLE_IMAGE_ENCODER: bool = True
    IMAGE_MODEL_NAME: str = "WinKawaks/vit-tiny-patch16-224"

    # --- Verdict thresholds -----------------------------------------------
    # A classifier probability below this confidence is surfaced to users as
    # "uncertain" rather than a confident real/fake call.
    UNCERTAIN_CONFIDENCE_THRESHOLD: float = 0.60

    # --- Real-time verification (future stage) -----------------------------
    # Placeholder toggle for the "supplement local ML with external evidence"
    # stage described in the model-training repo's docs. Wire a real
    # search/fact-check API into services/verification_service.py and flip
    # this on when ready.
    ENABLE_REALTIME_VERIFICATION: bool = False

    # --- Feed / pagination -------------------------------------------------
    DEFAULT_FEED_LIMIT: int = 20
    MAX_FEED_LIMIT: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
