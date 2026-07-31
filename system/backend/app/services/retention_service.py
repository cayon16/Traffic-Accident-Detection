from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from app.config import Settings
from app.database import Database

from .storage_service import delete_dir_under, delete_file_under


SHORT_LIVED_SEGMENT_STATUSES = {"normal", "suspicious", "failed", "duplicate"}


class RetentionService:
    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database

    def cleanup_once(self) -> None:
        if not self.settings.retention_enabled:
            return
        self._cleanup_segments()
        self._cleanup_evidence()

    def _cleanup_segments(self) -> None:
        now = datetime.utcnow()
        for segment in self.database.list_segments(limit=100000):
            received_at = parse_datetime(str(segment.get("received_at", "")))
            if received_at is None:
                continue

            status = str(segment.get("status", ""))
            if status in SHORT_LIVED_SEGMENT_STATUSES:
                max_age = timedelta(hours=self.settings.normal_segment_retention_hours)
            elif status == "accident":
                max_age = timedelta(days=self.settings.accident_segment_retention_days)
            else:
                continue

            if now - received_at <= max_age:
                continue

            path = Path(str(segment.get("stored_path", "")))
            self._delete_file_under(path, self.settings.uploaded_segments_dir)

    def _cleanup_evidence(self) -> None:
        cutoff = datetime.utcnow() - timedelta(days=self.settings.evidence_retention_days)
        root = self.settings.storage_dir
        if not root.exists():
            return

        for event_dir in root.glob("*/*/*"):
            if not event_dir.is_dir():
                continue
            modified_at = datetime.utcfromtimestamp(event_dir.stat().st_mtime)
            if modified_at < cutoff:
                self._delete_dir_under(event_dir, root)

    def _delete_file_under(self, path: Path, root: Path) -> None:
        delete_file_under(path, root)

    def _delete_dir_under(self, path: Path, root: Path) -> None:
        delete_dir_under(path, root)


def parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
