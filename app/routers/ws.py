from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.websockets.manager import manager

router = APIRouter(tags=["realtime"])


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Connect here to receive `post_created` / `post_updated` events as they
    happen -- see websockets/manager.py for the payload shape and the
    frontend README section this replaces (the in-memory pub/sub in
    api.js)."""

    await manager.connect(websocket)
    try:
        while True:
            # We don't expect inbound messages, but need to keep the
            # connection alive and detect disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
