from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.state import get_runtime


router = APIRouter(prefix="/api/streams", tags=["streams"])


@router.get("/{camera_id}/snapshot")
def snapshot(camera_id: str):
    data = get_runtime().streams.snapshot_jpeg(camera_id)
    if data is None:
        raise HTTPException(status_code=404, detail="snapshot not available")
    return Response(content=data, media_type="image/jpeg")


