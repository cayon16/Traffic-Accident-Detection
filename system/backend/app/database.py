from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


class Database:
    def __init__(self, sqlite_path: Path):
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def init(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT UNIQUE NOT NULL,
                    name TEXT,
                    rtsp_url TEXT NOT NULL,
                    location TEXT,
                    enabled INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'offline',
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    camera_id TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    max_score REAL NOT NULL,
                    status TEXT NOT NULL,
                    clip_path TEXT,
                    thumbnail_path TEXT,
                    clip_start_time TEXT,
                    clip_end_time TEXT,
                    clip_start_seconds REAL,
                    clip_end_seconds REAL,
                    decision_reason TEXT,
                    admin_note TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS event_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    score REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_id TEXT UNIQUE NOT NULL,
                    camera_id TEXT NOT NULL,
                    original_filename TEXT,
                    stored_path TEXT NOT NULL,
                    segment_start TEXT NOT NULL,
                    segment_end TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    max_score REAL,
                    prediction TEXT,
                    event_id TEXT,
                    error TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                """
            )
            self._ensure_event_clip_columns()
            self._conn.commit()

    def _ensure_event_clip_columns(self) -> None:
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(events)").fetchall()
        }
        additions = {
            "clip_start_time": "TEXT",
            "clip_end_time": "TEXT",
            "clip_start_seconds": "REAL",
            "clip_end_seconds": "REAL",
        }
        for name, column_type in additions.items():
            if name not in columns:
                self._conn.execute(f"ALTER TABLE events ADD COLUMN {name} {column_type}")

    def upsert_camera(self, camera: dict[str, Any]) -> None:
        now = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cameras (camera_id, name, rtsp_url, location, enabled, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, COALESCE(?, 'offline'), ?, ?)
                ON CONFLICT(camera_id) DO UPDATE SET
                    name=excluded.name,
                    rtsp_url=excluded.rtsp_url,
                    location=excluded.location,
                    enabled=excluded.enabled,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (
                    camera["camera_id"],
                    camera.get("name", camera["camera_id"]),
                    camera["rtsp_url"],
                    camera.get("location", ""),
                    int(camera.get("enabled", True)),
                    camera.get("status"),
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def list_cameras(self) -> list[dict[str, Any]]:
        rows = self._query_all("SELECT * FROM cameras ORDER BY camera_id")
        return [dict(row) for row in rows]

    def get_camera(self, camera_id: str) -> dict[str, Any] | None:
        row = self._query_one("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,))
        return dict(row) if row else None

    def update_camera_status(self, camera_id: str, status: str) -> None:
        now = utc_now_iso()
        self._execute(
            "UPDATE cameras SET status = ?, updated_at = ? WHERE camera_id = ?",
            (status, now, camera_id),
        )

    def disable_cameras_not_in(self, camera_ids: list[str]) -> None:
        now = utc_now_iso()
        if not camera_ids:
            self._execute(
                "UPDATE cameras SET enabled = 0, status = 'offline', updated_at = ?",
                (now,),
            )
            return

        placeholders = ", ".join("?" for _ in camera_ids)
        self._execute(
            f"""
            UPDATE cameras
            SET enabled = 0, status = 'offline', updated_at = ?
            WHERE camera_id NOT IN ({placeholders})
            """,
            (now, *camera_ids),
        )

    def delete_camera(self, camera_id: str) -> None:
        self._execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))

    def create_segment(self, segment: dict[str, Any]) -> None:
        now = utc_now_iso()
        self._execute(
            """
            INSERT INTO segments (
                segment_id, camera_id, original_filename, stored_path,
                segment_start, segment_end, received_at, status,
                max_score, prediction, event_id, error, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                segment["segment_id"],
                segment["camera_id"],
                segment.get("original_filename"),
                segment["stored_path"],
                segment["segment_start"],
                segment["segment_end"],
                segment["received_at"],
                segment.get("status", "queued"),
                segment.get("max_score"),
                segment.get("prediction"),
                segment.get("event_id"),
                segment.get("error"),
                now,
                now,
            ),
        )

    def update_segment(
        self,
        segment_id: str,
        status: str,
        max_score: float | None = None,
        prediction: str | None = None,
        event_id: str | None = None,
        error: str | None = None,
    ) -> None:
        now = utc_now_iso()
        self._execute(
            """
            UPDATE segments
            SET status = ?, max_score = ?, prediction = ?, event_id = ?,
                error = ?, updated_at = ?
            WHERE segment_id = ?
            """,
            (status, max_score, prediction, event_id, error, now, segment_id),
        )

    def list_segments(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._query_all(
            "SELECT * FROM segments ORDER BY received_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in rows]

    def get_segment(self, segment_id: str) -> dict[str, Any] | None:
        row = self._query_one("SELECT * FROM segments WHERE segment_id = ?", (segment_id,))
        return dict(row) if row else None

    def create_event(self, event: dict[str, Any]) -> None:
        now = utc_now_iso()
        self._execute(
            """
            INSERT INTO events (
                event_id, camera_id, event_time, max_score, status,
                clip_path, thumbnail_path, decision_reason, admin_note,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_id"],
                event["camera_id"],
                event["event_time"],
                float(event["max_score"]),
                event.get("status", "pending"),
                event.get("clip_path"),
                event.get("thumbnail_path"),
                event.get("decision_reason"),
                event.get("admin_note"),
                now,
                now,
            ),
        )

    def update_event_status(self, event_id: str, status: str, admin_note: str = "") -> None:
        now = utc_now_iso()
        self._execute(
            "UPDATE events SET status = ?, admin_note = ?, updated_at = ? WHERE event_id = ?",
            (status, admin_note, now, event_id),
        )

    def update_event_clip(
        self,
        event_id: str,
        clip_path: str | None,
        thumbnail_path: str | None,
        clip_start_time: str | None = None,
        clip_end_time: str | None = None,
        clip_start_seconds: float | None = None,
        clip_end_seconds: float | None = None,
    ) -> None:
        now = utc_now_iso()
        self._execute(
            """
            UPDATE events
            SET clip_path = ?, thumbnail_path = ?,
                clip_start_time = COALESCE(?, clip_start_time),
                clip_end_time = COALESCE(?, clip_end_time),
                clip_start_seconds = COALESCE(?, clip_start_seconds),
                clip_end_seconds = COALESCE(?, clip_end_seconds),
                updated_at = ?
            WHERE event_id = ?
            """,
            (
                clip_path,
                thumbnail_path,
                clip_start_time,
                clip_end_time,
                clip_start_seconds,
                clip_end_seconds,
                now,
                event_id,
            ),
        )

    def mark_segment_storage_deleted(self, segment_id: str, reason: str) -> None:
        now = utc_now_iso()
        self._execute(
            """
            UPDATE segments
            SET error = TRIM(COALESCE(error, '') || ' ' || ?), updated_at = ?
            WHERE segment_id = ?
            """,
            (f"[storage deleted: {reason}]", now, segment_id),
        )

    def list_events(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._query_all(
            "SELECT * FROM events ORDER BY event_time DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in rows]

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        row = self._query_one("SELECT * FROM events WHERE event_id = ?", (event_id,))
        return dict(row) if row else None

    def add_event_scores(self, event_id: str, timestamps: list[str], scores: list[float]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO event_scores (event_id, timestamp, score) VALUES (?, ?, ?)",
                [(event_id, ts, float(score)) for ts, score in zip(timestamps, scores)],
            )
            self._conn.commit()

    def get_event_scores(self, event_id: str) -> list[dict[str, Any]]:
        rows = self._query_all(
            "SELECT timestamp, score FROM event_scores WHERE event_id = ? ORDER BY timestamp",
            (event_id,),
        )
        return [dict(row) for row in rows]

    def delete_all_events(self) -> None:
        now = utc_now_iso()
        with self._lock:
            self._conn.execute("DELETE FROM event_scores")
            self._conn.execute("DELETE FROM events")
            self._conn.execute(
                """
                UPDATE segments
                SET event_id = NULL, updated_at = ?
                WHERE event_id IS NOT NULL
                """,
                (now,),
            )
            self._conn.commit()

    def has_event_near(self, camera_id: str, event_time: str, window_seconds: float) -> bool:
        try:
            target = datetime.fromisoformat(event_time)
        except ValueError:
            return False
        rows = self._query_all(
            "SELECT event_time FROM events WHERE camera_id = ? ORDER BY event_time DESC LIMIT 100",
            (camera_id,),
        )
        for row in rows:
            try:
                existing = datetime.fromisoformat(str(row["event_time"]))
            except ValueError:
                continue
            if abs((target - existing).total_seconds()) <= window_seconds:
                return True
        return False

    def _execute(self, sql: str, params: tuple[Any, ...]) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def _query_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())
