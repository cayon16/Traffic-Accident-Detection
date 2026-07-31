from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import Settings

from .decision_manager import Decision


class EvidenceArchive:
    def __init__(self, settings: Settings):
        self.settings = settings

    def write_package(
        self,
        event_id: str,
        camera_id: str,
        segment: dict[str, Any],
        decision: Decision,
        clip_path: Path,
        thumbnail_path: Path | None,
        source_segment_path: Path,
        clip_window: dict[str, Any] | None = None,
    ) -> None:
        package_dir = clip_path.parent
        package_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "event_id": event_id,
            "camera_id": camera_id,
            "segment_id": segment.get("segment_id"),
            "original_filename": segment.get("original_filename"),
            "source_segment_path": str(source_segment_path),
            "segment_start": segment.get("segment_start"),
            "segment_end": segment.get("segment_end"),
            "event_time": decision.event_time,
            "max_score": decision.max_score,
            "max_score_index": decision.max_score_index,
            "decision_reason": decision.decision_reason,
            "threshold_warning": self.settings.threshold_warning,
            "threshold_accident": self.settings.threshold_accident,
            "frame_stride": self.settings.frame_stride,
            "segment_seconds": self.settings.segment_seconds,
            "segment_interval_seconds": self.settings.segment_interval_seconds,
            "pre_event_seconds": self.settings.pre_event_seconds,
            "post_event_seconds": self.settings.post_event_seconds,
            "model_backend": self.settings.model_backend,
            "score_source": self.settings.score_source,
            "clip_onnx_path": self.settings.clip_onnx_path,
            "vad_onnx_path": self.settings.vad_onnx_path,
            "clip_path": str(clip_path),
            "thumbnail_path": str(thumbnail_path) if thumbnail_path else "",
            "created_at": datetime.utcnow().replace(microsecond=0).isoformat(),
        }
        if clip_window:
            metadata.update(clip_window)
        write_json(package_dir / "metadata.json", metadata)
        write_scores(package_dir / "scores.csv", decision.score_timestamps, decision.scores)
        write_text(package_dir / "source_segment_ref.txt", str(source_segment_path) + "\n")
        write_hashes(
            package_dir / "sha256.txt",
            {
                "accident_clip.mp4": clip_path,
                "thumbnail.jpg": thumbnail_path,
                "source_segment": source_segment_path,
            },
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_scores(path: Path, timestamps: list[str], scores: list[float]) -> None:
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "score"])
        for timestamp, score in zip(timestamps, scores):
            writer.writerow([timestamp, f"{float(score):.8f}"])
    temporary_path.replace(path)


def write_text(path: Path, text: str) -> None:
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def write_hashes(path: Path, files: dict[str, Path | None]) -> None:
    lines = []
    for label, file_path in files.items():
        if file_path is None or not file_path.exists():
            continue
        lines.append(f"{sha256_file(file_path)}  {label}")
    write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
