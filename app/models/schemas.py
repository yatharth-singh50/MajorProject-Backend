"""
Pydantic schemas.

These mirror the shapes the frontend already expects (see
`MajorProject-Code/src/services/mockData.js` and `api.js`) so that swapping
`services/api.js`'s function bodies for `fetch()` calls, per the frontend
README, requires no shape changes on that side.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_.]+$")
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    identifier: str = Field(min_length=1, description="Username or email")
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

class UserOut(BaseModel):
    id: str
    username: str
    displayName: str
    bio: str = ""
    location: str = ""
    avatarColor: str = "#D97757"
    avatarImage: Optional[str] = None  # data: URI, or null to fall back to the initials circle
    bannerImage: Optional[str] = None  # data: URI, or null to fall back to the gradient banner
    joinedAt: Optional[str] = None
    platformVerified: bool = False
    # "admin" (gold tick + Sathi icon) | "gold" | "news" (red) | "government"
    # (green) | "company" (black/white) | null. See sql_models.VALID_TIERS.
    verificationTier: Optional[str] = None
    isAdmin: bool = False
    languages: List[str] = []
    followerCount: int = 0
    followingCount: int = 0
    trustScore: Optional[int] = None  # 0-100 "credibility ring", or null if no fact-checked posts yet
    trustCount: int = 0  # how many fact-checked posts the score is based on


class UserUpdate(BaseModel):
    """PATCH /users/{username} body. All fields optional -- only supplied
    fields are changed, mirroring `updateUser(username, patch)` on the
    frontend, which does `Object.assign(user, patch)`."""

    display_name: Optional[str] = Field(default=None, alias="displayName")
    bio: Optional[str] = None
    location: Optional[str] = None
    avatar_color: Optional[str] = Field(default=None, alias="avatarColor")
    # Raw data: URI strings -- validated/size-checked in routers/users.py via
    # utils/media.py, same as post images, rather than in this schema.
    avatar_image: Optional[str] = Field(default=None, alias="avatarImage")
    banner_image: Optional[str] = Field(default=None, alias="bannerImage")
    languages: Optional[List[str]] = None
    auto_analyze: Optional[bool] = Field(default=None, alias="autoAnalyze")
    disputed_threshold: Optional[int] = Field(default=None, alias="disputedThreshold", ge=0, le=100)
    default_post_language: Optional[str] = Field(default=None, alias="defaultPostLanguage")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# ML pipeline / analysis
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    real = "real"
    fake = "fake"
    uncertain = "uncertain"


class AnalysisStatus(str, Enum):
    processing = "processing"
    analyzed = "analyzed"
    failed = "failed"
    skipped = "skipped"  # replies -- the pipeline only ever runs on top-level posts


class LanguageOut(BaseModel):
    code: str
    name: str


class MatchedClaim(BaseModel):
    title: str
    source: str
    stance: str  # "supports" | "contradicts"
    url: Optional[str] = None


class PipelineStage(BaseModel):
    stage: str
    label: str
    detail: str = ""


class VerificationStatus(str, Enum):
    """Factual verification outcome -- deliberately a SEPARATE signal from
    `Verdict` above. `Verdict` is what the MuRIL text classifier predicts;
    this is what (if any) real-time evidence retrieval concluded. The
    frontend must not collapse these into one "verified real" stamp -- see
    AI ARCHITECTURE section 10 of the project spec."""

    supported = "supported"
    contradicted = "contradicted"
    mixed = "mixed"
    insufficient = "insufficient"
    unavailable = "unavailable"  # no evidence stage ran (disabled, or provider unreachable)


class ExtractedClaim(BaseModel):
    """Output of the claim-extraction stage (qwen2.5:3b locally, Groq in the
    cloud) -- the specific checkable claim pulled out of the post, plus a
    suggested search query for the (future) evidence-retrieval stage."""

    claim: Optional[str] = None
    entities: List[str] = []
    searchQuery: Optional[str] = None
    provider: str = "none"  # "ollama" | "groq" | "none"


class ImageUnderstanding(BaseModel):
    """Output of the image-understanding stage (MiniCPM-V locally, Gemini in
    the cloud) -- OCR'd text and any claim the image appears to make. This
    is NOT a real/fake verdict on the image -- see image_encoder.py."""

    ocrText: Optional[str] = None
    claim: Optional[str] = None
    provider: str = "none"  # "ollama" | "gemini" | "none"


class OverallAssessment(BaseModel):
    """A DERIVED, third signal -- computed by combining the model
    classification and factual verification below with explicit precedence
    rules (see ml_pipeline.py::_compute_overall_assessment), not by letting
    either one silently overwrite the other. When live evidence
    contradicts/supports a claim, it dominates this field regardless of how
    confident the text classifier was, because it's checked against
    current reality rather than being a style/pattern judgment. When no
    evidence was gathered (or it was inconclusive), this just mirrors the
    model classification, capped at a lower confidence since it's unverified."""

    label: Verdict
    confidence: float
    reason: str


class AnalysisOut(BaseModel):
    # --- Preserved for frontend compatibility -------------------------------
    status: AnalysisStatus
    verdict: Optional[Verdict] = None
    confidence: Optional[float] = None
    model: str
    explanation: str = ""
    matchedClaims: List[MatchedClaim] = []
    pipeline: List[PipelineStage] = []

    # --- Extended: model classification vs. factual verification, kept
    # explicitly separate per the project spec. Do not infer one from the
    # other in the UI. ---------------------------------------------------
    extractedClaim: Optional[ExtractedClaim] = None
    imageUnderstanding: Optional[ImageUnderstanding] = None
    verificationStatus: VerificationStatus = VerificationStatus.unavailable
    aiMode: Optional[str] = None  # "local" | "hybrid" | "cloud" -- which mode produced this analysis
    overallAssessment: Optional[OverallAssessment] = None


class AnalyzeRequest(BaseModel):
    """Body for POST /analyze -- a direct call into the ML pipeline,
    independent of posting. Matches the shape `mlService.js.runPipeline()`
    is called with, plus an optional image."""

    text: str = Field(min_length=1, max_length=10_000)
    image_base64: Optional[str] = None
    image_mime_type: Optional[str] = None


class AnalyzeResponse(BaseModel):
    verdict: Verdict
    confidence: float
    model: str
    language: LanguageOut
    explanation: str
    matchedClaims: List[MatchedClaim] = []
    pipeline: List[PipelineStage] = []
    image_analysis: Optional[dict] = None
    extractedClaim: Optional[ExtractedClaim] = None
    imageUnderstanding: Optional[ImageUnderstanding] = None
    verificationStatus: VerificationStatus = VerificationStatus.unavailable
    aiMode: Optional[str] = None
    overallAssessment: Optional[OverallAssessment] = None


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------

class MediaOut(BaseModel):
    mimeType: str
    # Base64-encoded image bytes. See utils/media.py -- this is the
    # "translate images into byte code" storage approach requested for the
    # prototype; swap for an object-storage URL later without touching the
    # response shape (just populate `url` instead / additionally).
    dataBase64: Optional[str] = None
    url: Optional[str] = None


class AttachmentOut(BaseModel):
    """One piece of media on a post. Uploaded images/videos are stored as raw
    bytes in Mongo and served from `/media/{id}` (so <img>/<video> can load
    them with Range support for seeking); GIFs from the GIF picker are just
    an external URL. `dataBase64` only ever appears on legacy posts created
    before attachments existed."""

    id: Optional[str] = None
    kind: str  # "image" | "video" | "gif"
    mimeType: str
    url: str = ""
    dataBase64: Optional[str] = None


class StatsOut(BaseModel):
    likes: int = 0
    reposts: int = 0
    comments: int = 0
    views: int = 0


class PostCreate(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    languageCode: Optional[str] = None
    parentId: Optional[str] = None
    image_base64: Optional[str] = Field(default=None, description="Optional raw base64 image payload")
    image_mime_type: Optional[str] = None
    # A GIF picked from the frontend's GIF search (Klipy) -- a URL, not an
    # upload. Deliberately a SEPARATE field from image_base64: GIFs never
    # produce `raw_image_bytes` for the pipeline, so they never reach MiniCPM-V
    # /ViT/Gemini. Per the project's own scope: only real images contribute to
    # fake-news/context analysis, not GIFs (decorative) or other attachments.
    gif_url: Optional[str] = None
    # New multi-attachment path: ids returned by POST /media/upload, plus any
    # number of picked GIF URLs (total capped by MAX_ATTACHMENTS).
    mediaIds: List[str] = []
    gifUrls: List[str] = []
    # The "News" tag. Only posts tagged as news (or containing an obvious
    # news keyword like #breaking / "breaking news") go through the
    # fact-check pipeline; everything else is treated as an ordinary post.
    isNews: bool = False


class PostOut(BaseModel):
    id: str
    authorId: str
    author: Optional[UserOut] = None
    parentId: Optional[str] = None
    language: LanguageOut
    content: str
    translation: Optional[str] = None
    media: Optional[MediaOut] = None  # legacy single-media field
    attachments: List[AttachmentOut] = []
    isNews: bool = False
    createdAt: str
    stats: StatsOut
    likedByMe: bool = False
    repostedByMe: bool = False
    analysis: AnalysisOut


class ThreadOut(BaseModel):
    """A post's conversation context: the chain of posts above it (root
    first) and every reply beneath it at any depth, oldest first. The
    client nests replies using each reply's `parentId`."""

    ancestors: List[PostOut]
    replies: List[PostOut]


class VerificationUpdate(BaseModel):
    tier: Optional[str] = None  # one of sql_models.VALID_TIERS, or null to revoke


class FeedResponse(BaseModel):
    posts: List[PostOut]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TrendingTag(BaseModel):
    tag: str
    posts: int
    language: str


class SearchResponse(BaseModel):
    posts: List[PostOut]
    users: List[UserOut]


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

class NotificationOut(BaseModel):
    id: str
    type: str  # "like" | "repost" | "reply" | "mention" | "verdict_ready"
    actorId: Optional[str] = None
    actor: Optional[UserOut] = None
    postId: Optional[str] = None
    snippet: Optional[str] = None  # first part of the post the notification is about
    message: str
    read: bool = False
    createdAt: str


TokenResponse.model_rebuild()
