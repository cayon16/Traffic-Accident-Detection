from __future__ import annotations

from typing import Any

from app.config import Settings
from app.database import Database

from .segment_service import SegmentService
from .stream_worker import StreamWorker


class StreamManager:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        segment_service: SegmentService,
    ):
        self.settings = settings
        self.database = database
        self.segment_service = segment_service
        self._workers: dict[str, StreamWorker] = {}

    def start_enabled(self) -> None:
        for camera in self.database.list_cameras():
            if int(camera.get("enabled", 0)) == 1:
                self.start_camera(camera)

    def start_camera(self, camera: dict[str, Any]) -> None:
        camera_id = str(camera["camera_id"])
        worker = self._workers.get(camera_id)
        if worker is None:
            worker = StreamWorker(
                camera,
                self.settings,
                self.database,
                self.segment_service,
            )
            self._workers[camera_id] = worker
        worker.start()

    def stop_camera(self, camera_id: str) -> None:
        worker = self._workers.get(camera_id)
        if worker:
            worker.stop()

    def stop_all(self) -> None:
        for worker in list(self._workers.values()):
            worker.stop()

    def snapshot_jpeg(self, camera_id: str) -> bytes | None:
        worker = self._workers.get(camera_id)
        if worker is None:
            return None
        return worker.snapshot_jpeg()
