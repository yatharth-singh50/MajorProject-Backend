# Sathi Backend

FastAPI backend for the [Sāthi frontend prototype](https://github.com/yatharth-singh50/MajorProject-Code)
— a Twitter-like social app with an automated, multilingual fact-checking
pipeline built on the [model-training repo](https://github.com/yatharth-singh148/major-project-model-training)'s
MuRIL classifier.

This backend implements exactly the REST contract the frontend's
`src/services/api.js` already expects (see that repo's README, "Wiring up
your real backend") plus auth, so you can swap the mock store for real
`fetch()` calls with no shape changes on the frontend side.

## Architecture

- **FastAPI** — async API layer.
- **Postgres (Supabase-compatible)** — accounts/auth. Relational, unique
  usernames/emails, password hashes. Swap `POSTGRES_URI` for a Supabase
  connection string when ready; local Postgres works identically for dev.
- **MongoDB** — posts, replies, likes/reposts, notifications, and each
  post's ML analysis payload. Chosen for the flexible/nested shape of that
  data (media, `matchedClaims`, pipeline stages) and read-heavy feed access.
- **Images** — accepted as base64 (or a `data:` URI), validated, and stored
  as byte-string data directly on the Mongo post document. Swap
  `app/utils/media.py` for object storage (S3 / Supabase Storage) later
  without touching any response shapes.
- **ML pipeline** (`app/services/`) — language ID → cheap heuristic triage →
  MuRIL transformer classification → optional image feature extraction →
  stubbed real-time verification. See "ML pipeline" below.
- **WebSocket** (`GET /ws`) — broadcasts `post_created` / `post_updated`
  events, replacing the frontend's in-memory pub/sub with real push.

```
backend/
  app/
    core/        # settings, JWT + password hashing
    db/          # Mongo (motor) + Postgres (SQLAlchemy async) connections
    models/      # SQLAlchemy ORM (users) + Pydantic schemas (API shapes)
    crud/        # DB read/write logic
    routers/     # HTTP route handlers
    services/    # the ML pipeline (language ID, classifier, image, verification)
    websockets/  # connection manager for live post updates
    utils/       # image validation/decoding
  requirements.txt
  .env.example
  docker-compose.yml   # local Mongo + Postgres for dev
```

## Setup

1. **Start databases** (or point at your own / Supabase):
   ```bash
   docker compose up -d
   ```

2. **Install dependencies:**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   The heavy ML packages (`torch`, `transformers`, etc.) are commented out
   at the bottom of `requirements.txt`. The API runs and stays fully
   functional without them — see "ML pipeline" below for what happens
   instead. Uncomment and install them once you have the trained checkpoint.

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # edit .env: at minimum set a real JWT_SECRET_KEY
   ```

4. **Run:**
   ```bash
   uvicorn app.main:app --reload
   ```
   Tables are created automatically on startup (via SQLAlchemy
   `create_all` — fine for this stage; move to Alembic migrations before
   any real production rollout). Interactive API docs: `http://localhost:8000/docs`.

5. **Point the frontend at it:** in `MajorProject-Code`, replace the bodies
   of the functions in `src/services/api.js` with `fetch()` calls to this
   API per its own README table, and `src/services/mlService.js`'s
   `runPipeline()` with a call to `POST /analyze`.

## Endpoint map

Matches the frontend README's suggested mapping exactly, plus auth and a
few extensions:

| Frontend function | Method & path |
| --- | --- |
| — | `POST /auth/register` |
| — | `POST /auth/login` |
| `getCurrentUser()` | `GET /auth/me` |
| `getFeed({ limit })` | `GET /feed?limit=` |
| `getPost(id)` | `GET /posts/:id` |
| `getReplies(postId)` | `GET /posts/:id/replies` |
| `createPost(...)` | `POST /posts` |
| `toggleLike(postId)` | `POST /posts/:id/like` |
| `toggleRepost(postId)` | `POST /posts/:id/repost` |
| `getUserByUsername(username)` | `GET /users/:username` |
| `getUserPosts(username, tab)` | `GET /users/:username/posts?tab=` |
| `updateUser(username, patch)` | `PATCH /users/:username` |
| `getTrending()` | `GET /trending` |
| `searchAll(query)` | `GET /search?q=` |
| `mlService.runPipeline(text)` | `POST /analyze` |
| — | `GET /notifications`, `POST /notifications/:id/read` |
| — (pub/sub replacement) | `WS /ws` |

All endpoints return the same field names/shapes as the frontend's
`mockData.js`/`api.js` (`camelCase`, `stats.{likes,reposts,comments,views}`,
`analysis.{status,verdict,confidence,model,explanation,matchedClaims,pipeline}`,
etc.) — see `app/models/schemas.py` for the authoritative shapes.

Auth: send `Authorization: Bearer <access_token>` (from `/auth/login` or
`/auth/register`) on any endpoint that needs a current user. Read endpoints
that only *optionally* care who's asking (feed, a post, search) work
without a token too — `likedByMe`/`repostedByMe` just come back `false`.

## ML pipeline

`app/services/ml_pipeline.py` runs, per post:

1. **`lang_id`** — Unicode-script heuristics for the Indic languages the
   frontend already names (Hindi, Tamil, Bengali, Gujarati, Malayalam), with
   `langdetect` as a general fallback so languages beyond that fixed list
   (and beyond English) are still identified rather than defaulting to
   English.
2. **`small_lm`** — a cheap heuristic (not a trained model) that filters out
   opinions/questions before spending a full classifier pass on them, so
   personal statements don't get a misleading real/fake stamp.
3. **`transformer`** — the MuRIL checkpoint from the training repo
   (`app/services/text_classifier.py`, loaded exactly as that repo's own
   `src/predict.py` does). **If `TEXT_MODEL_PATH` doesn't exist, or
   `torch`/`transformers` aren't installed, this transparently falls back to
   a lightweight heuristic classifier** so the rest of the system (routing,
   storage, auth, the API contract, the frontend) is fully exercisable
   without the multi-GB model files. Check `GET /health` to see which
   backend is active (`"muril"` or `"heuristic"`).
4. **`image`** (optional, only if a post includes one) — a frozen ViT-tiny
   feature extractor, mirroring the training repo's `demo_multimodal.py`.
   **This does not affect the verdict yet** — there's no trained multimodal
   fusion model in the training repo yet (its own docs: "Multimodal fusion:
   Planned / next implementation stage"), so this stage only reports that
   an image was encoded. `app/services/image_encoder.py` is the seam to plug
   a real fusion classifier into once one exists.
5. **`verification`** — stubbed (`app/services/verification_service.py`),
   returns no claims unless `ENABLE_REALTIME_VERIFICATION=true` and you've
   implemented `fetch_evidence()` against a real search/fact-check API. This
   matches the training repo's docs, which describe real-time verification
   as a "planned backend component" and explicitly warn that ML prediction
   and factual verification are separate signals — a post isn't "false"
   just because the classifier says FAKE, or because no evidence was found.

Verdict mapping: MuRIL/the heuristic returns `REAL`/`FAKE` + a confidence.
If confidence is below `UNCERTAIN_CONFIDENCE_THRESHOLD` (default `0.60`),
the verdict surfaced to users is `"uncertain"` rather than a confident call.

## Wiring in the real trained model

1. Copy the `./models/muril_combined` checkpoint directory from the
   training repo's output somewhere this API can read.
2. Set `TEXT_MODEL_PATH` in `.env` to that path.
3. `pip install torch transformers` (uncomment them in `requirements.txt`).
4. Restart the API — `GET /health` should report `"text_classifier_backend": "muril"`.

No other code changes needed; `app/services/text_classifier.py` handles the
rest.

## Notes / deliberate scope decisions

- **Multimodal fusion and real-time verification are intentionally not
  implemented** — the training repo's own documentation says not to claim
  results for either until they've actually been trained/built. Both have
  clearly marked extension points (`image_encoder.py`, `verification_service.py`)
  instead of being faked.
- **Table/collection creation is automatic** (`create_all` / Mongo index
  creation on startup) to keep first-run friction at zero. Move to Alembic
  migrations before this goes anywhere near production data.
- **Trust score / "credibility ring"** is computed the same way the
  frontend's mock does: `real / (real + fake)` over a user's analyzed posts,
  ignoring `uncertain` ones, `null` if they have none yet.
