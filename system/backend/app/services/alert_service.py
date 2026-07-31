from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket


class AlertService:
    def __init__(self):
        self._connections: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    def publish(self, payload: dict[str, Any]) -> None:
        if self._loop is None or not self._connections:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        message = json.dumps(payload, ensure_ascii=False)
        dead: list[WebSocket] = []
        for websocket in list(self._connections):
            try:
                await websocket.send_text(message)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(websocket)


