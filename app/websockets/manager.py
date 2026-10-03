"""
Minimal WebSocket broadcast manager.

The frontend's mock `api.js` uses an in-memory pub/sub (`onPostCreated` /
`onPostUpdated`) so any mounted list re-renders when a post is created or
its analysis finishes -- the README explicitly calls out swapping that for
"websocket/SSE push updates from your backend" wired to the same two emit
points. This is that server side: connect to `GET /ws`, and you'll receive
JSON messages shaped like:

    {"type": "post_created", "post": {...PostOut...}}
    {"type": "post_updated", "post": {...PostOut...}}

No auth/rooms/filtering yet -- every connected client gets every event,
which is fine for a single global feed. Add per-user filtering here if/when
you introduce private posts or DMs.
"""

import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self._connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self._connections:
            return
        payload = json.dumps(message, default=str)
        dead: list[WebSocket] = []
        for connection in self._connections:
            try:
                await connection.send_text(payload)
            except Exception:  # noqa: BLE001
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)

    async def broadcast_post_created(self, post: dict) -> None:
        await self.broadcast({"type": "post_created", "post": post})

    async def broadcast_post_updated(self, post: dict) -> None:
        await self.broadcast({"type": "post_updated", "post": post})

    async def broadcast_post_deleted(self, post_ids: list[str]) -> None:
        await self.broadcast({"type": "post_deleted", "ids": post_ids})


manager = ConnectionManager()
