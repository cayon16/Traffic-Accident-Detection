import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from clip import clip
from crop import image_crop


PRESETS: Dict[str, Dict[str, str]] = {
    "custom": {"crop_indices": "", "label_mode": "none"},
    "fivecrop": {"crop_indices": "0,1,2,3,4", "label_mode": "parent"},
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_crop_indices(crop_text: str) -> List[int]:
    if crop_text.strip() == "":
        return []
    indices: List[int] = []
    for token in crop_text.split(","):
        token = token.strip()
        if token == "":
            continue
        idx = int(token)
        if idx < 0 or idx > 4:
            raise ValueError("crop index must be in [0, 4] for the five-crop pipeline")
        indices.append(idx)
    if len(indices) == 0:
        raise ValueError("crop indices are empty")
    return indices


def resolve_setting(args: argparse.Namespace, key: str) -> str:
    if getattr(args, key):
        return getattr(args, key)
    return PRESETS[args.preset][key]


def list_videos_from_file(video_root: Path, video_list: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    with video_list.open("r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if rel == "" or rel.startswith("#"):
                continue
            rel_path = Path(rel.replace("\\", "/"))
            if rel_path.is_absolute():
                video_path = rel_path
                try:
                    rel_path = rel_path.relative_to(video_root)
                except ValueError:
                    rel_path = Path(rel_path.name)
            else:
                video_path = video_root / rel_path
            pairs.append((video_path, rel_path))
    return pairs


def list_videos_from_dir(video_root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for video_path in sorted(video_root.rglob("*")):
        if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTS:
            rel_path = video_path.relative_to(video_root)
            pairs.append((video_path, rel_path))
    return pairs


def pick_frame(chunk: Sequence[np.ndarray], sample_position: str) -> np.ndarray:
    if sample_position == "first":
        idx = 0
    elif sample_position == "middle":
        idx = len(chunk) // 2
    else:
        idx = len(chunk) - 1
    return chunk[idx]


def sample_video_frames(video_path: Path, sample_stride: int, sample_position: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    sampled: List[np.ndarray] = []
    chunk: List[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        chunk.append(frame)
        if len(chunk) == sample_stride:
            sampled.append(pick_frame(chunk, sample_position))
            chunk = []

    if len(chunk) > 0:
        sampled.append(pick_frame(chunk, sample_position))

    cap.release()

    if len(sampled) == 0:
        raise RuntimeError(f"video has no readable frames: {video_path}")

    return sampled


def encode_crop(
    frames: Sequence[np.ndarray],
    crop_idx: int,
    model: torch.nn.Module,
    preprocess,
    device: str,
    batch_size: int,
) -> np.ndarray:
    feat_list: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start:start + batch_size]
            batch_tensors: List[torch.Tensor] = []
            for frame in batch_frames:
                crop_rgb = image_crop(frame, crop_idx)
                pil_img = Image.fromarray(crop_rgb)
                batch_tensors.append(preprocess(pil_img))

            image_batch = torch.stack(batch_tensors, dim=0).to(device)
            feat = model.encode_image(image_batch)
            feat_list.append(feat.float().cpu().numpy())

    return np.concatenate(feat_list, axis=0).astype(np.float32)


def infer_label(rel_path: Path, label_mode: str) -> str:
    if label_mode == "none":
        return ""
    if label_mode == "parent":
        if len(rel_path.parts) == 1:
            return rel_path.parent.name
        return rel_path.parts[0]
    raise ValueError(f"unsupported label mode: {label_mode}")


def build_output_path(output_root: Path, rel_path: Path, crop_idx: int) -> Path:
    stem_rel = rel_path.with_suffix("")
    out_dir = output_root / stem_rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem_rel.name}__{crop_idx}.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CLIP ViT-B/16 features in VadCLIP-compatible format"
    )
    parser.add_argument("--video-root", required=True, help="root directory of input videos")
    parser.add_argument("--output-root", required=True, help="root directory to save .npy features")
    parser.add_argument(
        "--video-list",
        default="",
        help="optional txt file of relative paths (e.g., list/Anomaly_Train.txt)",
    )
    parser.add_argument(
        "--csv-output",
        default="",
        help="optional path to write path,label csv in VadCLIP style",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="custom",
        help="dataset split preset for crop and label settings",
    )
    parser.add_argument(
        "--crop-indices",
        default="",
        help="comma-separated crop ids in [0..4], overrides preset if set",
    )
    parser.add_argument(
        "--label-mode",
        choices=["", "none", "parent"],
        default="",
        help="label parsing mode for csv, overrides preset if set",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=16,
        help="temporal stride in frames; one feature is extracted per stride chunk",
    )
    parser.add_argument(
        "--sample-position",
        choices=["first", "middle", "last"],
        default="first",
        help="which frame to pick from each stride chunk",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="CLIP image batch size")
    parser.add_argument("--model-name", default="ViT-B/16", help="CLIP model name")
    parser.add_argument("--download-root", default="", help="optional CLIP checkpoint cache directory")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="inference device",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a video when all target crop files already exist",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="debug option: only process first N videos (0 means all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sample_stride <= 0:
        raise ValueError("sample_stride must be positive")

    crop_text = resolve_setting(args, "crop_indices")
    label_mode = resolve_setting(args, "label_mode")

    crop_indices = parse_crop_indices(crop_text)
    if len(crop_indices) == 0:
        raise ValueError("no crop index is configured; set --preset or --crop-indices")

    video_root = Path(args.video_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.video_list:
        video_list = Path(args.video_list).expanduser().resolve()
        video_pairs = list_videos_from_file(video_root, video_list)
    else:
        video_pairs = list_videos_from_dir(video_root)

    if args.max_videos > 0:
        video_pairs = video_pairs[: args.max_videos]

    if len(video_pairs) == 0:
        raise RuntimeError("no videos found")

    print(f"Loading CLIP model {args.model_name} on {args.device}...")
    model, preprocess = clip.load(
        args.model_name,
        device=args.device,
        download_root=(args.download_root if args.download_root else None),
    )
    model.eval()

    csv_rows: List[Tuple[str, str]] = []

    for idx, (video_path, rel_path) in enumerate(video_pairs, start=1):
        if not video_path.exists():
            print(f"[{idx}/{len(video_pairs)}] missing: {video_path}")
            continue

        out_paths = [build_output_path(output_root, rel_path, c) for c in crop_indices]
        if args.skip_existing and all(p.exists() for p in out_paths):
            print(f"[{idx}/{len(video_pairs)}] skip existing: {rel_path.as_posix()}")
            if args.csv_output and label_mode != "none":
                label = infer_label(rel_path, label_mode)
                for p in out_paths:
                    csv_rows.append((p.resolve().as_posix(), label))
            continue

        try:
            frames = sample_video_frames(video_path, args.sample_stride, args.sample_position)
        except Exception as exc:
            print(f"[{idx}/{len(video_pairs)}] failed reading video {rel_path.as_posix()}: {exc}")
            continue

        print(
            f"[{idx}/{len(video_pairs)}] {rel_path.as_posix()} -> "
            f"{len(frames)} temporal features x {len(crop_indices)} crops"
        )

        for crop_idx, out_path in zip(crop_indices, out_paths):
            feat = encode_crop(frames, crop_idx, model, preprocess, args.device, args.batch_size)
            np.save(out_path, feat)

            if args.csv_output and label_mode != "none":
                label = infer_label(rel_path, label_mode)
                csv_rows.append((out_path.resolve().as_posix(), label))

    if args.csv_output:
        csv_path = Path(args.csv_output).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            for path_str, label in csv_rows:
                writer.writerow([path_str, label])
        print(f"Saved csv: {csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()
