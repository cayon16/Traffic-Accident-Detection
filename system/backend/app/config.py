from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SYSTEM_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = SYSTEM_DIR / "backend"
CONFIG_DIR = BACKEND_DIR / "configs"


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    name: str
    rtsp_url: str
    location: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    database_url: str
    storage_dir: Path
    thumbnails_dir: Path
    uploaded_segments_dir: Path
    keep_non_accident_segments: bool
    segment_worker_count: int
    segment_upload_api_key: str
    retention_enabled: bool
    normal_segment_retention_hours: float
    normal_segment_retention_days: int
    accident_segment_retention_days: int
    evidence_retention_days: int
    buffer_seconds: float
    segment_seconds: float
    segment_interval_seconds: float
    inference_interval_seconds: float
    inference_fps: float
    frame_stride: int
    reconnect_interval_seconds: float
    realtime_playback: bool
    loop_file_sources: bool
    threshold_warning: float
    threshold_accident: float
    min_consecutive_hits: int
    smoothing_window: int
    cooldown_seconds: float
    pre_event_seconds: float
    post_event_seconds: float
    clip_boundary_threshold: float
    clip_boundary_buffer_seconds: float
    clip_boundary_max_gap_seconds: float
    clip_output_fps: float
    model_backend: str
    pth_model_path: str
    pth_device: str
    clip_download_root: str
    vad_onnx_path: str
    clip_onnx_path: str
    score_source: str
    onnx_provider: str
    inference_mode: int
    spatial_top_k: int
    visual_length: int
    classes_num: int
    clip_batch_size: int


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        try:
            data = yaml.safe_load(file) or {}
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML file: {path}. If you use a Windows path, write it "
                "with forward slashes like C:/Users/... or wrap backslash paths in "
                "single quotes."
            ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _path_from_config(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path) if path else CONFIG_DIR / "settings.yaml"
    raw = _read_yaml(config_path)

    stream = raw.get("stream", {})
    decision = raw.get("decision", {})
    clip = raw.get("clip", {})
    database = raw.get("database", {})
    model = raw.get("model", {})
    storage = raw.get("storage", {})
    processing = raw.get("processing", {})
    security = raw.get("security", {})
    retention = raw.get("retention", {})

    storage_dir = _path_from_config(
        storage.get("evidence_dir", storage.get("events_dir", "storage/evidence")),
        BACKEND_DIR,
    )
    thumbnails_dir = _path_from_config(
        storage.get("thumbnails_dir", "storage/thumbnails"),
        BACKEND_DIR,
    )
    uploaded_segments_dir = _path_from_config(
        storage.get("uploaded_segments_dir", "storage/segments"),
        BACKEND_DIR,
    )

    return Settings(
        raw=raw,
        database_url=database.get("url", "sqlite:///accident_system.db"),
        storage_dir=storage_dir,
        thumbnails_dir=thumbnails_dir,
        uploaded_segments_dir=uploaded_segments_dir,
        keep_non_accident_segments=bool(storage.get("keep_non_accident_segments", False)),
        segment_worker_count=max(1, int(processing.get("segment_worker_count", 1))),
        segment_upload_api_key=str(security.get("segment_upload_api_key", "")),
        retention_enabled=bool(retention.get("enabled", False)),
        normal_segment_retention_hours=float(
            retention.get(
                "normal_segment_hours",
                float(retention.get("normal_segment_days", 7)) * 24.0,
            )
        ),
        normal_segment_retention_days=max(1, int(retention.get("normal_segment_days", 7))),
        accident_segment_retention_days=max(1, int(retention.get("accident_segment_days", 30))),
        evidence_retention_days=max(1, int(retention.get("evidence_days", 365))),
        buffer_seconds=float(stream.get("buffer_seconds", 60)),
        segment_seconds=float(stream.get("segment_seconds", stream.get("buffer_seconds", 60))),
        segment_interval_seconds=float(stream.get("segment_interval_seconds", stream.get("inference_interval_seconds", 50))),
        inference_interval_seconds=float(stream.get("inference_interval_seconds", 5)),
        inference_fps=float(stream.get("inference_fps", 4)),
        frame_stride=max(1, int(stream.get("frame_stride", 16))),
        reconnect_interval_seconds=float(stream.get("reconnect_interval_seconds", 5)),
        realtime_playback=bool(stream.get("realtime_playback", True)),
        loop_file_sources=bool(stream.get("loop_file_sources", True)),
        threshold_warning=float(decision.get("threshold_warning", 0.70)),
        threshold_accident=float(decision.get("threshold_accident", 0.80)),
        min_consecutive_hits=int(decision.get("min_consecutive_hits", 2)),
        smoothing_window=int(decision.get("smoothing_window", 3)),
        cooldown_seconds=float(decision.get("cooldown_seconds", 60)),
        pre_event_seconds=float(clip.get("pre_event_seconds", 10)),
        post_event_seconds=float(clip.get("post_event_seconds", 10)),
        clip_boundary_threshold=float(
            clip.get("boundary_threshold", decision.get("threshold_accident", 0.80))
        ),
        clip_boundary_buffer_seconds=float(clip.get("boundary_buffer_seconds", 3)),
        clip_boundary_max_gap_seconds=float(clip.get("boundary_max_gap_seconds", 2)),
        clip_output_fps=float(clip.get("output_fps", 25)),
        model_backend=str(model.get("backend", "mock")),
        pth_model_path=(
            str(_path_from_config(str(model.get("pth_model_path", "")), BACKEND_DIR))
            if model.get("pth_model_path")
            else ""
        ),
        pth_device=str(model.get("pth_device", "auto")),
        clip_download_root=(
            str(_path_from_config(str(model.get("clip_download_root", "")), BACKEND_DIR))
            if model.get("clip_download_root")
            else ""
        ),
        vad_onnx_path=(
            str(_path_from_config(str(model.get("vad_onnx_path", "")), BACKEND_DIR))
            if model.get("vad_onnx_path")
            else ""
        ),
        clip_onnx_path=(
            str(_path_from_config(str(model.get("clip_onnx_path", "")), BACKEND_DIR))
            if model.get("clip_onnx_path")
            else ""
        ),
        score_source=str(model.get("score_source", "fusion")),
        onnx_provider=str(model.get("onnx_provider", "auto")),
        inference_mode=int(model.get("inference_mode", 2)),
        spatial_top_k=int(model.get("spatial_top_k", 3)),
        visual_length=int(model.get("visual_length", 256)),
        classes_num=int(model.get("classes_num", 2)),
        clip_batch_size=int(model.get("clip_batch_size", 64)),
    )


def load_camera_configs(path: str | Path | None = None) -> list[CameraConfig]:
    config_path = Path(path) if path else CONFIG_DIR / "cameras.yaml"
    raw = _read_yaml(config_path)
    cameras = raw.get("cameras", [])
    if not isinstance(cameras, list):
        raise ValueError("cameras.yaml must contain a 'cameras' list")

    result: list[CameraConfig] = []
    for item in cameras:
        if not isinstance(item, dict):
            raise ValueError("each camera config must be a mapping")
        result.append(
            CameraConfig(
                camera_id=str(item["camera_id"]),
                name=str(item.get("name", item["camera_id"])),
                rtsp_url=str(item["rtsp_url"]),
                location=str(item.get("location", "")),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return result


def sqlite_path_from_url(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("MVP backend currently supports sqlite:/// URLs only")
    raw_path = database_url[len(prefix):]
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return BACKEND_DIR / path
