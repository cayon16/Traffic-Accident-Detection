from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from app.database import Database

from .alert_service import AlertService
from .decision_manager import Decision


class EventService:
    def __init__(self, database: Database, alerts: AlertService):
        self.database = database
        self.alerts = alerts

    def create_pending_event(self, camera_id: str, decision: Decision) -> str:
        event_time = decision.event_time or datetime.utcnow().replace(microsecond=0).isoformat()
        event_id = self._new_event_id()
        self.database.create_event(
            {
                "event_id": event_id,
                "camera_id": camera_id,
                "event_time": event_time,
                "max_score": decision.max_score,
                "status": "pending",
                "decision_reason": decision.decision_reason,
            }
        )
        self.database.add_event_scores(
            event_id,
            decision.score_timestamps,
            decision.scores,
        )
        self.alerts.publish(self._alert_payload("ACCIDENT_ALERT", event_id))
        return event_id

    def update_clip(
        self,
        event_id: str,
        clip_path: str,
        thumbnail_path: str | None,
        clip_start_time: str | None = None,
        clip_end_time: str | None = None,
        clip_start_seconds: float | None = None,
        clip_end_seconds: float | None = None,
    ) -> None:
        self.database.update_event_clip(
            event_id,
            clip_path,
            thumbnail_path,
            clip_start_time,
            clip_end_time,
            clip_start_seconds,
            clip_end_seconds,
        )
        self.alerts.publish(self._alert_payload("EVENT_UPDATED", event_id))

    def update_status(self, event_id: str, status: str, admin_note: str = "") -> dict[str, Any]:
        self.database.update_event_status(event_id, status, admin_note)
        self.alerts.publish(self._alert_payload("EVENT_UPDATED", event_id))
        event = self.database.get_event(event_id)
        if event is None:
            raise RuntimeError(f"event not found after status update: {event_id}")
        return event

    def _alert_payload(self, alert_type: str, event_id: str) -> dict[str, Any]:
        event = self.database.get_event(event_id)
        if event is None:
            return {"type": alert_type, "event_id": event_id}
        camera = self.database.get_camera(event["camera_id"]) or {}
        return {
            "type": alert_type,
            "event_id": event["event_id"],
            "camera_id": event["camera_id"],
            "camera_name": camera.get("name", event["camera_id"]),
            "event_time": event["event_time"],
            "max_score": event["max_score"],
            "clip_url": f"/api/events/{event_id}/clip",
            "status": event["status"],
        }

    def _new_event_id(self) -> str:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"EVT_{stamp}_{uuid4().hex[:8]}"
