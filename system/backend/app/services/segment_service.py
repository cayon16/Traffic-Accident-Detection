from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import UploadFile

from app.config import Settings
from app.database import Database

from .clip_extractor import ClipExtractor
from .decision_manager import DecisionManager
from .evidence_archive import EvidenceArchive
from .event_service import EventService
from .storage_service import NON_ACCIDENT_STATUSES, delete_file_under
from .vadclip_service import VADCLIPService


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class ClipWindow:
    start_seconds: float
    end_seconds: float
    threshold: float
    buffer_seconds: float
    start_index: int
    end_index: int
    reason: str

    @property
    def duration_seconds(self) -> float:
        return max(0.1, self.end_seconds - self.start_seconds)

    def to_metadata(self) -> dict:
        return {
            "clip_start_seconds": self.start_seconds,
            "clip_end_seconds": self.end_seconds,
            "clip_duration_seconds": self.duration_seconds,
            "clip_boundary_threshold": self.threshold,
            "clip_boundary_buffer_seconds": self.buffer_seconds,
            "clip_start_score_index": self.start_index,
            "clip_end_score_index": self.end_index,
            "clip_selection_reason": self.reason,
        }


class SegmentService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        detector: VADCLIPService,
        decision_manager: DecisionManager,
        event_service: EventService,
        clip_extractor: ClipExtractor,
    ):
        self.settings = settings
        self.database = database
        self.detector = detector
        self.decision_manager = decision_manager
        self.event_service = event_service
        self.clip_extractor = clip_extractor
        self.evidence_archive = EvidenceArchive(settings)
        self._predict_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []

    def start(self) -> None:
        if any(worker.is_alive() for worker in self._workers):
            return
        self._stop_event.clear()
        self._workers = []
        for index in range(self.settings.segment_worker_count):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"SegmentWorker-{index + 1}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def stop(self) -> None:
        self._stop_event.set()
        for _ in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join(timeout=2)

    def enqueue_segment(self, segment_id: str) -> None:
        self._queue.put(segment_id)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            segment_id = self._queue.get()
            try:
                if segment_id is None:
                    return
                self.process_segment(str(segment_id))
            finally:
                self._queue.task_done()

    async def accept_upload(
        self,
        camera_id: str,
        upload: UploadFile,
        segment_start: str = "",
        segment_end: str = "",
    ) -> dict:
        filename = upload.filename or "segment.mp4"
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise ValueError(f"unsupported video extension: {suffix}")

        received_at = datetime.utcnow().replace(microsecond=0)
        segment_id = self._new_segment_id(camera_id, received_at)
        output_dir = self.settings.uploaded_segments_dir / camera_id / received_at.strftime("%Y-%m-%d")
        output_dir.mkdir(parents=True, exist_ok=True)
        stored_path = output_dir / f"{segment_id}{suffix}"

        with stored_path.open("wb") as file:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)

        duration_seconds, _, _ = read_video_metadata(stored_path)
        start_time, end_time = resolve_segment_times(
            segment_start,
            segment_end,
            received_at,
            duration_seconds,
        )

        self._ensure_camera(camera_id)
        self.database.create_segment(
            {
                "segment_id": segment_id,
                "camera_id": camera_id,
                "original_filename": filename,
                "stored_path": str(stored_path),
                "segment_start": start_time.isoformat(),
                "segment_end": end_time.isoformat(),
                "received_at": received_at.isoformat(),
                "status": "queued",
            }
        )
        return self.database.get_segment(segment_id) or {"segment_id": segment_id}

    def process_segment(self, segment_id: str) -> None:
        segment = self.database.get_segment(segment_id)
        if segment is None:
            return

        self.database.update_segment(segment_id, "processing")
        camera_id = str(segment["camera_id"])
        source_path = Path(str(segment["stored_path"]))

        try:
            segment_start = parse_datetime(str(segment["segment_start"]))
            segment_end = parse_datetime(str(segment["segment_end"]))
            segment_duration = max(0.1, (segment_end - segment_start).total_seconds())
            frames, timestamps = sample_video_for_model(
                source_path,
                segment_start,
                self.settings.frame_stride,
            )
            with self._predict_lock:
                prediction = self.detector.predict(frames, timestamps)
            decision = self.decision_manager.evaluate(camera_id, prediction)

            if not decision.is_accident:
                status = "suspicious" if decision.is_suspicious else "normal"
                self.database.update_segment(
                    segment_id,
                    status,
                    decision.max_score,
                    "normal",
                    error=decision.decision_reason,
                )
                self._delete_source_segment_if_needed(segment_id, source_path, status)
                return

            event_time = parse_datetime(decision.event_time) if decision.event_time else segment_start
            duplicate_window = max(
                self.settings.pre_event_seconds,
                self.settings.post_event_seconds,
                self.settings.segment_seconds - self.settings.segment_interval_seconds,
            )
            if self.database.has_event_near(camera_id, event_time.isoformat(), duplicate_window):
                self.database.update_segment(
                    segment_id,
                    "duplicate",
                    decision.max_score,
                    "accident",
                    error="event already exists near max-score time",
                )
                self._delete_source_segment_if_needed(segment_id, source_path, "duplicate")
                return

            event_id = self.event_service.create_pending_event(camera_id, decision)
            clip_window = select_accident_clip_window(
                decision,
                segment_start,
                event_time,
                segment_duration,
                self.settings.clip_boundary_threshold,
                self.settings.clip_boundary_buffer_seconds,
                self.settings.clip_boundary_max_gap_seconds,
                self.settings.pre_event_seconds,
                self.settings.post_event_seconds,
            )
            clip_path, thumbnail_path = self.clip_extractor.extract_from_video(
                camera_id,
                event_id,
                event_time,
                source_path,
                clip_window.start_seconds,
                clip_window.duration_seconds,
            )
            self.evidence_archive.write_package(
                event_id,
                camera_id,
                segment,
                decision,
                clip_path,
                thumbnail_path,
                source_path,
                clip_window.to_metadata(),
            )
            clip_start_time = segment_start + timedelta(seconds=clip_window.start_seconds)
            clip_end_time = segment_start + timedelta(seconds=clip_window.end_seconds)
            self.event_service.update_clip(
                event_id,
                str(clip_path),
                str(thumbnail_path),
                clip_start_time.replace(microsecond=0).isoformat(),
                clip_end_time.replace(microsecond=0).isoformat(),
                clip_window.start_seconds,
                clip_window.end_seconds,
            )
            self.database.update_segment(
                segment_id,
                "accident",
                decision.max_score,
                "accident",
                event_id=event_id,
            )
        except Exception as exc:
            self.database.update_segment(segment_id, "failed", error=str(exc))
            self._delete_source_segment_if_needed(segment_id, source_path, "failed")
            print(f"[{camera_id}] segment processing failed for {segment_id}: {exc}")

    def _ensure_camera(self, camera_id: str) -> None:
        camera = self.database.get_camera(camera_id)
        if camera is None:
            self.database.upsert_camera(
                {
                    "camera_id": camera_id,
                    "name": camera_id,
                    "rtsp_url": f"upload://{camera_id}",
                    "location": "Segment upload",
                    "enabled": True,
                    "status": "online",
                }
            )
        else:
            self.database.upsert_camera(
                {
                    "camera_id": camera_id,
                    "name": camera.get("name", camera_id),
                    "rtsp_url": camera.get("rtsp_url") or f"upload://{camera_id}",
                    "location": camera.get("location", "Segment upload"),
                    "enabled": True,
                    "status": "online",
                }
            )

    def _new_segment_id(self, camera_id: str, received_at: datetime) -> str:
        stamp = received_at.strftime("%Y%m%d_%H%M%S")
        source = "".join(ch if ch.isalnum() else "_" for ch in camera_id)
        return f"SEG_{source}_{stamp}_{uuid4().hex[:8]}"

    def _delete_source_segment_if_needed(self, segment_id: str, source_path: Path, status: str) -> None:
        if self.settings.keep_non_accident_segments:
            return
        if status not in NON_ACCIDENT_STATUSES:
            return
        if delete_file_under(source_path, self.settings.uploaded_segments_dir):
            self.database.mark_segment_storage_deleted(
                segment_id,
                f"{status} segment auto-deleted",
            )


def read_video_metadata(video_path: Path) -> tuple[float, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid video metadata: {video_path}")
    return frame_count / fps, fps, frame_count


def sample_video_for_model(
    video_path: Path,
    segment_start: datetime,
    frame_stride: int,
) -> tuple[list, list[datetime]]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"invalid FPS for video: {video_path}")

    frames = []
    timestamps: list[datetime] = []
    frame_id = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_id % frame_stride == 0:
            frames.append(frame)
            timestamps.append(segment_start + timedelta(seconds=frame_id / fps))
        frame_id += 1
    capture.release()

    if not frames:
        raise RuntimeError(f"video has no readable frames: {video_path}")
    return frames, timestamps


def select_accident_clip_window(
    decision,
    segment_start: datetime,
    fallback_event_time: datetime,
    segment_duration: float,
    threshold: float,
    buffer_seconds: float,
    max_gap_seconds: float,
    fallback_pre_seconds: float,
    fallback_post_seconds: float,
) -> ClipWindow:
    scores = [float(score) for score in decision.scores]
    timestamps = [parse_datetime(str(value)) for value in decision.score_timestamps]
    if len(scores) != len(timestamps) or not scores:
        return fallback_clip_window(
            segment_start,
            fallback_event_time,
            segment_duration,
            threshold,
            buffer_seconds,
            fallback_pre_seconds,
            fallback_post_seconds,
            "fallback: missing score timeline",
        )

    runs = merge_close_runs(
        contiguous_score_runs(scores, threshold),
        timestamps,
        max_gap_seconds,
    )
    if not runs:
        return fallback_clip_window(
            segment_start,
            fallback_event_time,
            segment_duration,
            threshold,
            buffer_seconds,
            fallback_pre_seconds,
            fallback_post_seconds,
            "fallback: no score run over boundary threshold",
        )

    score_step = estimate_score_step_seconds(timestamps)
    best_run = None
    best_rank = None
    for start_index, end_index in runs:
        start_seconds = max(0.0, (timestamps[start_index] - segment_start).total_seconds())
        end_seconds = max(
            start_seconds,
            (timestamps[end_index] - segment_start).total_seconds() + score_step,
        )
        run_duration = max(0.0, end_seconds - start_seconds)
        run_peak = max(scores[start_index:end_index + 1])
        rank = (run_duration, run_peak, -start_index)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_run = (start_index, end_index, start_seconds, end_seconds)

    start_index, end_index, start_seconds, end_seconds = best_run
    clip_start = max(0.0, start_seconds - buffer_seconds)
    clip_end = min(segment_duration, end_seconds + buffer_seconds)
    if clip_end <= clip_start:
        clip_end = min(segment_duration, clip_start + max(0.1, score_step))

    return ClipWindow(
        start_seconds=clip_start,
        end_seconds=clip_end,
        threshold=threshold,
        buffer_seconds=buffer_seconds,
        start_index=start_index,
        end_index=end_index,
        reason="longest contiguous score run over boundary threshold",
    )


def contiguous_score_runs(scores: list[float], threshold: float) -> list[tuple[int, int]]:
    runs = []
    start_index = None
    for index, score in enumerate(scores):
        if score >= threshold:
            if start_index is None:
                start_index = index
        elif start_index is not None:
            runs.append((start_index, index - 1))
            start_index = None
    if start_index is not None:
        runs.append((start_index, len(scores) - 1))
    return runs


def merge_close_runs(
    runs: list[tuple[int, int]],
    timestamps: list[datetime],
    max_gap_seconds: float,
) -> list[tuple[int, int]]:
    if not runs or max_gap_seconds <= 0:
        return runs

    merged = [runs[0]]
    for start_index, end_index in runs[1:]:
        previous_start, previous_end = merged[-1]
        gap_seconds = (timestamps[start_index] - timestamps[previous_end]).total_seconds()
        if gap_seconds <= max_gap_seconds:
            merged[-1] = (previous_start, end_index)
        else:
            merged.append((start_index, end_index))
    return merged


def estimate_score_step_seconds(timestamps: list[datetime]) -> float:
    deltas = [
        (timestamps[index + 1] - timestamps[index]).total_seconds()
        for index in range(len(timestamps) - 1)
    ]
    deltas = sorted(delta for delta in deltas if delta > 0)
    if not deltas:
        return 0.0
    return deltas[len(deltas) // 2]


def fallback_clip_window(
    segment_start: datetime,
    event_time: datetime,
    segment_duration: float,
    threshold: float,
    buffer_seconds: float,
    pre_seconds: float,
    post_seconds: float,
    reason: str,
) -> ClipWindow:
    event_offset = max(0.0, (event_time - segment_start).total_seconds())
    start_seconds = max(0.0, event_offset - pre_seconds)
    end_seconds = min(segment_duration, event_offset + post_seconds)
    if end_seconds <= start_seconds:
        end_seconds = min(segment_duration, start_seconds + max(0.1, pre_seconds + post_seconds))
    return ClipWindow(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        threshold=threshold,
        buffer_seconds=buffer_seconds,
        start_index=-1,
        end_index=-1,
        reason=reason,
    )


def resolve_segment_times(
    segment_start: str,
    segment_end: str,
    received_at: datetime,
    duration_seconds: float,
) -> tuple[datetime, datetime]:
    start_time = parse_datetime(segment_start) if segment_start else None
    end_time = parse_datetime(segment_end) if segment_end else None

    if start_time is None and end_time is None:
        end_time = received_at
        start_time = end_time - timedelta(seconds=duration_seconds)
    elif start_time is None and end_time is not None:
        start_time = end_time - timedelta(seconds=duration_seconds)
    elif start_time is not None and end_time is None:
        end_time = start_time + timedelta(seconds=duration_seconds)

    return start_time.replace(microsecond=0), end_time.replace(microsecond=0)


def parse_datetime(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed.replace(microsecond=0)
