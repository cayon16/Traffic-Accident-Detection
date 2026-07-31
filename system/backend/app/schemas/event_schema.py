from __future__ import annotations

from pydantic import BaseModel


class StatusUpdate(BaseModel):
    admin_note: str = ""


class EventOut(BaseModel):
    event_id: str
    camera_id: str
    event_time: str
    max_score: float
    status: str
    clip_path: str | None = None
    thumbnail_path: str | None = None
    decision_reason: str | None = None
    admin_note: str | None = None


