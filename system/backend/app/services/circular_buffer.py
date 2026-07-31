from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock

import numpy as np


@dataclass
class FrameItem:
    frame: np.ndarray
    timestamp: datetime
    frame_id: int


class CircularBuffer:
    def __init__(self, buffer_seconds: float):
        self.buffer_seconds = float(buffer_seconds)
        self._items: deque[FrameItem] = deque()
        self._lock = Lock()

    def append(self, frame: np.ndarray, timestamp: datetime, frame_id: int) -> FrameItem:
        item = FrameItem(frame=frame.copy(), timestamp=timestamp, frame_id=frame_id)
        cutoff = timestamp - timedelta(seconds=self.buffer_seconds)
        with self._lock:
            self._items.append(item)
            while self._items and self._items[0].timestamp < cutoff:
                self._items.popleft()
        return item

    def recent(self, seconds: float) -> list[FrameItem]:
        cutoff = datetime.utcnow() - timedelta(seconds=seconds)
        with self._lock:
            return [item for item in self._items if item.timestamp >= cutoff]

    def window(self, start: datetime, end: datetime) -> list[FrameItem]:
        with self._lock:
            return [item for item in self._items if start <= item.timestamp <= end]

    def latest(self) -> FrameItem | None:
        with self._lock:
            return self._items[-1] if self._items else None

    def duration_seconds(self) -> float:
        with self._lock:
            if len(self._items) < 2:
                return 0.0
            return (self._items[-1].timestamp - self._items[0].timestamp).total_seconds()

