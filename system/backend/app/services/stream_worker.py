from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from app.config import Settings
from app.database import Database

from .circular_buffer import CircularBuffer
from .segment_service import SegmentService


class StreamWorker:
    def __init__(
        self,
        camera: dict[str, Any],
        settings: Settings,
        database: Database,
        segment_service: SegmentService,
    ):
        self.camera = camera
        self.settings = settings
        self.database = database
        self.segment_service = segment_service
        self.buffer = CircularBuffer(settings.buffer_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_id = 0
        self._source_fps = settings.clip_output_fps
        self._last_segment_sent_at = 0.0

    @property
    def camera_id(self) -> str:
        return str(self.camera["camera_id"])

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"StreamWorker-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def snapshot_jpeg(self) -> bytes | None:
        latest = self.buffer.latest()
        if latest is None:
            return None
        ok, encoded = cv2.imencode(".jpg", latest.frame)
        return encoded.tobytes() if ok else None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            source = parse_video_source(str(self.camera["rtsp_url"]))
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.database.update_camera_status(self.camera_id, "reconnecting")
                time.sleep(self.settings.reconnect_interval_seconds)
                continue

            self.database.update_camera_status(self.camera_id, "online")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = self.settings.clip_output_fps
            self._source_fps = float(fps)
            frame_sleep = 1.0 / fps if self.settings.realtime_playback else 0.0

            try:
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break

                    now = datetime.utcnow()
                    self.buffer.append(frame, now, self._frame_id)
                    self._frame_id += 1

                    current_time = time.monotonic()
                    if current_time - self._last_segment_sent_at >= self.settings.segment_interval_seconds:
                        if self._send_recent_segment():
                            self._last_segment_sent_at = current_time

                    if frame_sleep > 0:
                        time.sleep(frame_sleep)
            finally:
                cap.release()

            if is_file_source(source) and self.settings.loop_file_sources:
                continue

            self.database.update_camera_status(self.camera_id, "reconnecting")
            time.sleep(self.settings.reconnect_interval_seconds)

        self.database.update_camera_status(self.camera_id, "offline")

    def _send_recent_segment(self) -> bool:
        if self.buffer.duration_seconds() < self.settings.segment_seconds:
            return False

        window = self.buffer.recent(self.settings.segment_seconds)
        if len(window) < 2:
            return False

        try:
            segment_id, segment_path = self._write_segment(window)
            segment_start = window[0].timestamp.replace(microsecond=0)
            segment_end = window[-1].timestamp.replace(microsecond=0)
            self.database.create_segment(
                {
                    "segment_id": segment_id,
                    "camera_id": self.camera_id,
                    "original_filename": f"{segment_id}.mp4",
                    "stored_path": str(segment_path),
                    "segment_start": segment_start.isoformat(),
                    "segment_end": segment_end.isoformat(),
                    "received_at": datetime.utcnow().replace(microsecond=0).isoformat(),
                    "status": "queued",
                }
            )
            self.segment_service.enqueue_segment(segment_id)
            print(
                f"[{self.camera_id}] queued segment {segment_id} "
                f"({self.settings.segment_seconds:g}s window every "
                f"{self.settings.segment_interval_seconds:g}s)"
            )
        except Exception as exc:
            print(f"[{self.camera_id}] segment upload skipped: {exc}")
            return True
        return True

    def _write_segment(self, frames: list[Any]) -> tuple[str, Path]:
        received_at = datetime.utcnow().replace(microsecond=0)
        source = "".join(ch if ch.isalnum() else "_" for ch in self.camera_id)
        segment_id = f"SEG_{source}_{received_at.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        output_dir = (
            self.settings.uploaded_segments_dir
            / self.camera_id
            / received_at.strftime("%Y-%m-%d")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{segment_id}.mp4"

        height, width = frames[0].frame.shape[:2]
        fps = self._estimate_segment_fps(frames)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer for {output_path}")
        for item in frames:
            writer.write(item.frame)
        writer.release()
        return segment_id, output_path

    def _estimate_segment_fps(self, frames: list[Any]) -> float:
        duration = max(0.1, (frames[-1].timestamp - frames[0].timestamp).total_seconds())
        fps = (len(frames) - 1) / duration if len(frames) > 1 else self._source_fps
        if fps <= 0:
            fps = self._source_fps
        if fps <= 0:
            fps = self.settings.clip_output_fps
        return float(fps)


def parse_video_source(value: str):
    if value.isdigit():
        return int(value)
    return value


def is_file_source(value) -> bool:
    if isinstance(value, int):
        return False
    return Path(str(value)).exists()
