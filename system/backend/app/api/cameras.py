from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.camera_schema import CameraIn
from app.state import get_runtime


router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("")
def list_cameras():
    return get_runtime().database.list_cameras()


@router.post("")
def create_camera(camera: CameraIn):
    runtime = get_runtime()
    runtime.database.upsert_camera(camera.model_dump())
    stored = runtime.database.get_camera(camera.camera_id)
    if stored and camera.enabled:
        runtime.streams.start_camera(stored)
    return stored


@router.put("/{camera_id}")
def update_camera(camera_id: str, camera: CameraIn):
    if camera_id != camera.camera_id:
        raise HTTPException(status_code=400, detail="camera_id mismatch")
    runtime = get_runtime()
    runtime.database.upsert_camera(camera.model_dump())
    stored = runtime.database.get_camera(camera_id)
    if stored and camera.enabled:
        runtime.streams.start_camera(stored)
    else:
        runtime.streams.stop_camera(camera_id)
    return stored


@router.delete("/{camera_id}")
def delete_camera(camera_id: str):
    runtime = get_runtime()
    runtime.streams.stop_camera(camera_id)
    runtime.database.delete_camera(camera_id)
    return {"ok": True}


@router.get("/{camera_id}/status")
def camera_status(camera_id: str):
    camera = get_runtime().database.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    return {"camera_id": camera_id, "status": camera.get("status", "offline")}


