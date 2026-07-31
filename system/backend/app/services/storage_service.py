from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import Settings
from app.database import Database


NON_ACCIDENT_STATUSES = {"normal", "suspicious", "failed", "duplicate"}


class StorageService:
    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database

    def summary(self) -> dict[str, Any]:
        segments = self.database.list_segments(limit=100000)
        events = self.database.list_events(limit=100000)
        status_counts: dict[str, int] = {}
        for segment in segments:
            status = str(segment.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "roots": {
                "segments": summarize_path(self.settings.uploaded_segments_dir),
                "evidence": summarize_path(self.settings.storage_dir),
                "thumbnails": summarize_path(self.settings.thumbnails_dir),
            },
            "segment_count": len(segments),
            "segment_status_counts": status_counts,
            "event_count": len(events),
            "events_with_clip": sum(1 for event in events if event.get("clip_path")),
        }

    def list_segments(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.database.list_segments(limit=limit)
        return [self._enrich_segment(row) for row in rows]

    def list_evidence(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.database.list_events(limit=limit)
        return [self._enrich_event(row) for row in rows]

    def delete_segment_file(self, segment_id: str, reason: str = "manual") -> dict[str, Any]:
        segment = self.database.get_segment(segment_id)
        if segment is None:
            raise KeyError("segment not found")

        path = Path(str(segment.get("stored_path", "")))
        deleted = delete_file_under(path, self.settings.uploaded_segments_dir)
        if deleted:
            self.database.mark_segment_storage_deleted(segment_id, reason)

        result = self._enrich_segment(segment)
        result["deleted"] = deleted
        return result

    def delete_event_evidence(self, event_id: str, reason: str = "manual") -> dict[str, Any]:
        event = self.database.get_event(event_id)
        if event is None:
            raise KeyError("event not found")

        clip_path = Path(str(event.get("clip_path") or ""))
        thumbnail_path = Path(str(event.get("thumbnail_path") or ""))
        deleted = False

        if clip_path:
            package_dir = clip_path.parent
            if is_under(package_dir, self.settings.storage_dir):
                deleted = delete_dir_under(package_dir, self.settings.storage_dir)
            else:
                deleted = delete_file_under(clip_path, self.settings.storage_dir)

        if thumbnail_path and thumbnail_path.exists():
            deleted = delete_file_under(thumbnail_path, self.settings.thumbnails_dir) or deleted

        if deleted:
            self.database.update_event_clip(event_id, None, None)
            self.database.update_event_status(event_id, "archived", f"storage deleted: {reason}")

        result = self._enrich_event(event)
        result["deleted"] = deleted
        return result

    def delete_all_events(self, reason: str = "manual reset") -> dict[str, Any]:
        events = self.database.list_events(limit=100000)
        deleted_packages = 0
        deleted_thumbnails = 0
        deleted_bytes = 0

        for event in events:
            clip_path = Path(str(event.get("clip_path") or ""))
            thumbnail_path = Path(str(event.get("thumbnail_path") or ""))

            if clip_path:
                package_dir = clip_path.parent
                if package_dir.is_dir() and is_under(package_dir, self.settings.storage_dir):
                    deleted_bytes += directory_size(package_dir)
                    if delete_dir_under(package_dir, self.settings.storage_dir):
                        deleted_packages += 1
                elif clip_path.is_file():
                    deleted_bytes += clip_path.stat().st_size
                    if delete_file_under(clip_path, self.settings.storage_dir):
                        deleted_packages += 1

            if thumbnail_path and thumbnail_path.is_file():
                if delete_file_under(thumbnail_path, self.settings.thumbnails_dir):
                    deleted_thumbnails += 1

        self.database.delete_all_events()
        return {
            "reason": reason,
            "event_count": len(events),
            "deleted_packages": deleted_packages,
            "deleted_thumbnails": deleted_thumbnails,
            "deleted_bytes": deleted_bytes,
        }

    def cleanup_segments_older_than(
        self,
        older_than_hours: float = 24.0,
        include_accident: bool = False,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        cutoff = datetime.utcnow() - timedelta(hours=max(0.0, older_than_hours))
        candidates = []

        for segment in self.database.list_segments(limit=100000):
            received_at = parse_datetime(str(segment.get("received_at", "")))
            if received_at is None or received_at > cutoff:
                continue
            status = str(segment.get("status", ""))
            if status == "accident" and not include_accident:
                continue
            if status not in NON_ACCIDENT_STATUSES and status != "accident":
                continue

            enriched = self._enrich_segment(segment)
            if enriched["file_exists"]:
                candidates.append(enriched)

        deleted = []
        if not dry_run:
            for item in candidates:
                result = self.delete_segment_file(
                    str(item["segment_id"]),
                    reason=f"older than {older_than_hours:g}h",
                )
                if result.get("deleted"):
                    deleted.append(result)

        return {
            "dry_run": dry_run,
            "older_than_hours": older_than_hours,
            "include_accident": include_accident,
            "candidate_count": len(candidates),
            "candidate_bytes": sum(int(item["file_size_bytes"]) for item in candidates),
            "deleted_count": len(deleted),
            "deleted_bytes": sum(int(item["file_size_bytes"]) for item in deleted),
            "candidates": candidates,
        }

    def _enrich_segment(self, segment: dict[str, Any]) -> dict[str, Any]:
        item = dict(segment)
        path = Path(str(item.get("stored_path", "")))
        item["file_exists"] = path.is_file()
        item["file_size_bytes"] = path.stat().st_size if path.is_file() else 0
        item["age_hours"] = age_hours(str(item.get("received_at", "")))
        return item

    def _enrich_event(self, event: dict[str, Any]) -> dict[str, Any]:
        item = dict(event)
        clip_path = Path(str(item.get("clip_path") or ""))
        package_dir = clip_path.parent if clip_path else Path()
        item["clip_exists"] = clip_path.is_file()
        item["package_dir"] = str(package_dir) if clip_path else ""
        item["package_exists"] = package_dir.is_dir() if clip_path else False
        item["package_size_bytes"] = directory_size(package_dir) if package_dir.is_dir() else 0
        return item


def summarize_path(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": count_files(path),
        "size_bytes": directory_size(path),
    }


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def delete_file_under(path: Path, root: Path) -> bool:
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
        if not resolved_path.is_file() or not resolved_path.is_relative_to(resolved_root):
            return False
        resolved_path.unlink()
        return True
    except OSError:
        return False


def delete_dir_under(path: Path, root: Path) -> bool:
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
        if not resolved_path.is_dir() or not resolved_path.is_relative_to(resolved_root):
            return False
        shutil.rmtree(resolved_path)
        return True
    except OSError:
        return False


def is_under(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def age_hours(value: str) -> float | None:
    parsed = parse_datetime(value)
    if parsed is None:
        return None
    return (datetime.utcnow() - parsed).total_seconds() / 3600.0
