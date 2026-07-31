from __future__ import annotations

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.api import cameras, events, streams, segments, storage
from app.config import SYSTEM_DIR, load_camera_configs, load_settings, sqlite_path_from_url
from app.database import Database
from app.services.alert_service import AlertService
from app.services.clip_extractor import ClipExtractor
from app.services.decision_manager import DecisionManager
from app.services.event_service import EventService
from app.services.retention_service import RetentionService
from app.services.segment_service import SegmentService
from app.services.storage_service import StorageService
from app.services.stream_manager import StreamManager
from app.services.vadclip_service import VADCLIPService
from app.state import RuntimeState, get_runtime, set_runtime


app = FastAPI(title="Traffic Accident Monitoring System")
app.include_router(cameras.router)
app.include_router(events.router)
app.include_router(streams.router)
app.include_router(segments.router)
app.include_router(storage.router)


@app.on_event("startup")
async def startup() -> None:
    settings = load_settings()
    database = Database(sqlite_path_from_url(settings.database_url))
    database.init()

    camera_configs = load_camera_configs()
    for camera in camera_configs:
        database.upsert_camera(camera.__dict__)
    database.disable_cameras_not_in([camera.camera_id for camera in camera_configs])

    alerts = AlertService()
    alerts.bind_loop(asyncio.get_running_loop())
    detector = VADCLIPService(settings)
    decision_manager = DecisionManager(
        settings.threshold_warning,
        settings.threshold_accident,
        settings.min_consecutive_hits,
        settings.smoothing_window,
        settings.cooldown_seconds,
    )
    clip_extractor = ClipExtractor(
        settings.storage_dir,
        settings.thumbnails_dir,
        settings.clip_output_fps,
    )
    event_service = EventService(database, alerts)
    RetentionService(settings, database).cleanup_once()
    storage_service = StorageService(settings, database)
    segment_service = SegmentService(
        settings,
        database,
        detector,
        decision_manager,
        event_service,
        clip_extractor,
    )
    stream_manager = StreamManager(
        settings,
        database,
        segment_service,
    )
    set_runtime(
        RuntimeState(
            settings=settings,
            database=database,
            alerts=alerts,
            events=event_service,
            segments=segment_service,
            storage=storage_service,
            streams=stream_manager,
        )
    )
    segment_service.start()
    stream_manager.start_enabled()


@app.on_event("shutdown")
async def shutdown() -> None:
    try:
        get_runtime().segments.stop()
        get_runtime().streams.stop_all()
    except RuntimeError:
        pass


@app.get("/")
def dashboard():
    index_path = SYSTEM_DIR / "frontend" / "index.html"
    if not index_path.exists():
        return {"message": "dashboard not found"}
    return FileResponse(index_path)


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    alerts = get_runtime().alerts
    await alerts.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        alerts.disconnect(websocket)
