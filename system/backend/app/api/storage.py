from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.state import get_runtime


router = APIRouter(prefix="/api/storage", tags=["storage"])


@router.get("/summary")
def storage_summary():
    return get_runtime().storage.summary()


@router.get("/segments")
def list_storage_segments(limit: int = 100):
    return get_runtime().storage.list_segments(limit=limit)


@router.get("/evidence")
def list_storage_evidence(limit: int = 100):
    return get_runtime().storage.list_evidence(limit=limit)


@router.delete("/segments/{segment_id}")
def delete_segment_file(segment_id: str):
    try:
        return get_runtime().storage.delete_segment_file(segment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/evidence/{event_id}")
def delete_event_evidence(event_id: str):
    try:
        return get_runtime().storage.delete_event_evidence(event_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/events")
def delete_all_events(confirm: str = ""):
    if confirm != "DELETE_ALL_EVENTS":
        raise HTTPException(
            status_code=400,
            detail="confirm=DELETE_ALL_EVENTS is required",
        )
    return get_runtime().storage.delete_all_events(reason="delete all events from UI")


@router.post("/cleanup/segments")
def cleanup_segments(
    older_than_hours: float = 24.0,
    include_accident: bool = False,
    dry_run: bool = True,
):
    return get_runtime().storage.cleanup_segments_older_than(
        older_than_hours=older_than_hours,
        include_accident=include_accident,
        dry_run=dry_run,
    )
