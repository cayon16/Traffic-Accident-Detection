from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.database import Database
from app.services.alert_service import AlertService
from app.services.event_service import EventService
from app.services.segment_service import SegmentService
from app.services.storage_service import StorageService
from app.services.stream_manager import StreamManager


@dataclass
class RuntimeState:
    settings: Settings
    database: Database
    alerts: AlertService
    events: EventService
    segments: SegmentService
    storage: StorageService
    streams: StreamManager


runtime: RuntimeState | None = None


def set_runtime(state: RuntimeState) -> None:
    global runtime
    runtime = state


def get_runtime() -> RuntimeState:
    if runtime is None:
        raise RuntimeError("runtime state has not been initialized")
    return runtime
