from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas.event_schema import StatusUpdate
from app.state import get_runtime


router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("")
def list_events(limit: int = 100):
    return get_runtime().database.list_events(limit=limit)


@router.get("/{event_id}")
def get_event(event_id: str):
    event = get_runtime().database.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    return event


@router.get("/{event_id}/clip")
def event_clip(event_id: str):
    event = get_runtime().database.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    clip_path = event.get("clip_path")
    if not clip_path or not Path(clip_path).exists():
        raise HTTPException(status_code=404, detail="clip not ready")
    return FileResponse(clip_path, media_type="video/mp4")


@router.get("/{event_id}/scores")
def event_scores(event_id: str):
    event = get_runtime().database.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    return get_runtime().database.get_event_scores(event_id)


@router.post("/{event_id}/confirm")
def confirm_event(event_id: str, payload: StatusUpdate | None = None):
    note = payload.admin_note if payload else ""
    return get_runtime().events.update_status(event_id, "confirmed", note)


@router.post("/{event_id}/false-alarm")
def false_alarm_event(event_id: str, payload: StatusUpdate | None = None):
    note = payload.admin_note if payload else ""
    return get_runtime().events.update_status(event_id, "false_alarm", note)


