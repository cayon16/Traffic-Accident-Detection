from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import cv2

from .circular_buffer import FrameItem


class ClipExtractor:
    def __init__(self, events_dir: Path, thumbnails_dir: Path, output_fps: float):
        self.events_dir = events_dir
        self.thumbnails_dir = thumbnails_dir
        self.output_fps = output_fps

    def extract(
        self,
        camera_id: str,
        event_id: str,
        event_time: datetime,
        frames: list[FrameItem],
        output_fps: float | None = None,
    ) -> tuple[Path, Path | None]:
        if not frames:
            raise RuntimeError("cannot extract clip from empty frame list")

        day_dir = event_time.strftime("%Y-%m-%d")
        clip_path, thumbnail_path = self._output_paths(camera_id, event_id, day_dir)
        fps = float(output_fps or self.output_fps)

        self._write_browser_mp4(clip_path, frames, fps)
        cv2.imwrite(str(thumbnail_path), frames[len(frames) // 2].frame)
        return clip_path, thumbnail_path

    def extract_from_video(
        self,
        camera_id: str,
        event_id: str,
        event_time: datetime,
        source_path: Path,
        start_seconds: float,
        duration_seconds: float,
    ) -> tuple[Path, Path | None]:
        day_dir = event_time.strftime("%Y-%m-%d")
        clip_path, thumbnail_path = self._output_paths(camera_id, event_id, day_dir)
        start_seconds = max(0.0, float(start_seconds))
        duration_seconds = max(0.1, float(duration_seconds))

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start_seconds:.3f}",
                    "-i",
                    str(source_path),
                    "-t",
                    f"{duration_seconds:.3f}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-an",
                    str(clip_path),
                ],
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            clip_path.unlink(missing_ok=True)
            self._write_video_range_opencv(
                source_path,
                clip_path,
                start_seconds,
                duration_seconds,
            )

        self._write_thumbnail(source_path, thumbnail_path, start_seconds + duration_seconds / 2.0)
        return clip_path, thumbnail_path

    def _output_paths(self, camera_id: str, event_id: str, day_dir: str) -> tuple[Path, Path]:
        package_dir = self.events_dir / camera_id / day_dir / event_id
        package_dir.mkdir(parents=True, exist_ok=True)
        return package_dir / "accident_clip.mp4", package_dir / "thumbnail.jpg"

    def _write_browser_mp4(self, clip_path: Path, frames: list[FrameItem], fps: float) -> None:
        temp_path = clip_path.with_suffix(".avi")
        self._write_opencv_video(temp_path, frames, fps, "MJPG")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(temp_path),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(clip_path),
                ],
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            clip_path.unlink(missing_ok=True)
            self._write_opencv_video(clip_path, frames, fps, "mp4v")
        finally:
            temp_path.unlink(missing_ok=True)

    def _write_opencv_video(
        self,
        path: Path,
        frames: list[FrameItem],
        fps: float,
        fourcc_name: str,
    ) -> None:
        height, width = frames[0].frame.shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer for {path}")
        for item in frames:
            writer.write(item.frame)
        writer.release()

    def _write_video_range_opencv(
        self,
        source_path: Path,
        clip_path: Path,
        start_seconds: float,
        duration_seconds: float,
    ) -> None:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open source video: {source_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = self.output_fps
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open video writer for {clip_path}")

        start_frame = max(0, int(start_seconds * fps))
        end_frame = max(start_frame + 1, int((start_seconds + duration_seconds) * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_id = start_frame
        while frame_id < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frame_id += 1

        writer.release()
        capture.release()

    def _write_thumbnail(self, source_path: Path, thumbnail_path: Path, timestamp_seconds: float) -> None:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            return
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_seconds) * 1000.0)
        ok, frame = capture.read()
        capture.release()
        if ok:
            cv2.imwrite(str(thumbnail_path), frame)
