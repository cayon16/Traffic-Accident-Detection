from __future__ import annotations

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from app.state import get_runtime


router = APIRouter(prefix="/api/segments", tags=["segments"])


@router.post("")
async def upload_segment(
    camera_id: str = Form(...),
    file: UploadFile = File(...),
    segment_start: str = Form(""),
    segment_end: str = Form(""),
    x_api_key: str = Header(""),
):
    runtime = get_runtime()
    expected_key = runtime.settings.segment_upload_api_key
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="invalid segment upload API key")

    try:
        segment = await runtime.segments.accept_upload(
            camera_id.strip(),
            file,
            segment_start,
            segment_end,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    runtime.segments.enqueue_segment(segment["segment_id"])
    return segment


@router.get("")
def list_segments(limit: int = 100):
    return get_runtime().database.list_segments(limit=limit)


@router.get("/{segment_id}")
def get_segment(segment_id: str):
    segment = get_runtime().database.get_segment(segment_id)
    if segment is None:
        raise HTTPException(status_code=404, detail="segment not found")
    return segment
