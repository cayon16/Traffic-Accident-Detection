from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Decision:
    is_accident: bool
    is_suspicious: bool
    event_time: str | None
    max_score: float
    max_score_index: int
    scores: list[float]
    score_timestamps: list[str]
    decision_reason: str


class DecisionManager:
    def __init__(
        self,
        threshold_warning: float,
        threshold_accident: float,
        min_consecutive_hits: int,
        smoothing_window: int,
        cooldown_seconds: float,
    ):
        self.threshold_warning = threshold_warning
        self.threshold_accident = threshold_accident
        self.min_consecutive_hits = max(1, min_consecutive_hits)
        self.smoothing_window = max(1, smoothing_window)
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_hits: dict[str, int] = {}
        self._last_event_time: dict[str, datetime] = {}

    def evaluate(self, camera_id: str, prediction: dict) -> Decision:
        raw_scores = [float(score) for score in prediction.get("scores", [])]
        timestamps = [str(ts) for ts in prediction.get("score_timestamps", [])]
        scores = self._smooth(raw_scores)
        if not scores:
            return Decision(False, False, None, 0.0, -1, [], [], "empty score series")

        max_index = max(range(len(scores)), key=lambda idx: scores[idx])
        max_score = float(scores[max_index])
        event_time = timestamps[max_index] if max_index < len(timestamps) else None

        if max_score < self.threshold_warning:
            self._consecutive_hits[camera_id] = 0
            return Decision(
                False,
                False,
                event_time,
                max_score,
                max_index,
                scores,
                timestamps,
                "max_score < threshold_warning",
            )

        if max_score < self.threshold_accident:
            self._consecutive_hits[camera_id] = 0
            return Decision(
                False,
                True,
                event_time,
                max_score,
                max_index,
                scores,
                timestamps,
                "threshold_warning <= max_score < threshold_accident",
            )

        hit_count = self._consecutive_hits.get(camera_id, 0) + 1
        self._consecutive_hits[camera_id] = hit_count
        if hit_count < self.min_consecutive_hits:
            return Decision(
                False,
                True,
                event_time,
                max_score,
                max_index,
                scores,
                timestamps,
                "accident threshold hit but consecutive requirement not satisfied",
            )

        now = datetime.utcnow()
        last_event_time = self._last_event_time.get(camera_id)
        if last_event_time is not None:
            elapsed = (now - last_event_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                return Decision(
                    False,
                    True,
                    event_time,
                    max_score,
                    max_index,
                    scores,
                    timestamps,
                    "camera is in cooldown",
                )

        self._last_event_time[camera_id] = now
        self._consecutive_hits[camera_id] = 0
        return Decision(
            True,
            True,
            event_time,
            max_score,
            max_index,
            scores,
            timestamps,
            "max_score >= threshold_accident and consecutive_hits satisfied",
        )

    def _smooth(self, scores: list[float]) -> list[float]:
        if self.smoothing_window <= 1:
            return scores
        smoothed: list[float] = []
        for idx in range(len(scores)):
            start = max(0, idx - self.smoothing_window + 1)
            values = scores[start:idx + 1]
            smoothed.append(sum(values) / len(values))
        return smoothed


