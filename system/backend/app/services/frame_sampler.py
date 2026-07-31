from __future__ import annotations

from datetime import datetime

from .circular_buffer import FrameItem


def sample_frames(items: list[FrameItem], target_fps: float) -> list[FrameItem]:
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if not items:
        return []

    min_interval = 1.0 / target_fps
    sampled: list[FrameItem] = []
    last_ts: datetime | None = None
    for item in items:
        if last_ts is None:
            sampled.append(item)
            last_ts = item.timestamp
            continue
        if (item.timestamp - last_ts).total_seconds() >= min_interval:
            sampled.append(item)
            last_ts = item.timestamp
    return sampled


def sample_frames_by_stride(items: list[FrameItem], frame_stride: int) -> list[FrameItem]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if not items:
        return []

    first_frame_id = items[0].frame_id
    return [
        item for item in items
        if (item.frame_id - first_frame_id) % frame_stride == 0
    ]

