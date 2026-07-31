from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crop import image_crop
from ucf_staged_training_common import split_feature


DEFAULT_INPUT_PATH = str(PROJECT_DIR / "data" / "sample_videos")
DEFAULT_OUTPUT_ROOT = str(PROJECT_DIR / "output" / "onnx_inference")
DEFAULT_ONNX_PATH = str(PROJECT_DIR / "models" / "vadclip_2class.onnx")
DEFAULT_CLIP_ONNX_PATH = str(PROJECT_DIR / "models" / "clip_vit_b16_image.onnx")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CROP_INDICES = (0, 1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-video or folder inference with CLIP ViT-B/16 ONNX + VADCLIP ONNX."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input video file or folder.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--onnx-model", default=DEFAULT_ONNX_PATH, help="VADCLIP ONNX model path.")
    parser.add_argument("--vad-onnx-model", default="", help="Alias for --onnx-model.")
    parser.add_argument("--clip-onnx-model", default=DEFAULT_CLIP_ONNX_PATH)
    parser.add_argument("--provider", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--frame-stride", type=int, default=16)
    parser.add_argument("--reference-fps", type=float, default=30.0)
    parser.add_argument("--sample-position", choices=["first", "middle", "last"], default="first")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--visual-length", type=int, default=256)
    parser.add_argument("--classes-num", type=int, default=2)
    parser.add_argument("--inference-mode", type=int, choices=[2], default=2)
    parser.add_argument("--spatial-top-k", type=int, default=3)
    parser.add_argument("--score-source", choices=["fusion", "a1", "a2"], default="a2")
    parser.add_argument("--suffix", default="", help="Output suffix. Empty uses _onnx_5crop.")
    parser.add_argument("--summary-name", default="onnx_results.csv")
    return parser.parse_args()


def convert_image_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def build_clip_preprocess(image_size: int = 224):
    return Compose([
        Resize(image_size, interpolation=BICUBIC),
        CenterCrop(image_size),
        convert_image_to_rgb,
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def sample_video_frames(
    video_path: Path,
    sample_stride: int,
    reference_fps: float,
    sample_position: str,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"invalid FPS for video: {video_path}")

    snippet_duration = sample_stride / reference_fps
    position_offset = {
        "first": 0.0,
        "middle": snippet_duration / 2.0,
        "last": max(0.0, snippet_duration - (1.0 / reference_fps)),
    }[sample_position]

    sampled: list[np.ndarray] = []
    previous_frame = None
    previous_time = 0.0
    last_frame = None
    frame_id = 0
    next_sample_time = position_offset

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_time = frame_id / fps
        while next_sample_time <= frame_time:
            if (
                previous_frame is not None
                and abs(previous_time - next_sample_time) < abs(frame_time - next_sample_time)
            ):
                sampled.append(previous_frame.copy())
            else:
                sampled.append(frame.copy())
            next_sample_time += snippet_duration

        previous_frame = frame
        previous_time = frame_time
        last_frame = frame
        frame_id += 1

    cap.release()

    expected_snippets = max(1, int(np.ceil((frame_id / fps) / snippet_duration)))
    while last_frame is not None and len(sampled) < expected_snippets:
        sampled.append(last_frame.copy())

    if len(sampled) == 0:
        raise RuntimeError(f"video has no readable frames: {video_path}")

    return sampled


def prepare_frame(frame: np.ndarray, mode: int, crop_idx: int) -> Image.Image:
    if mode != 2:
        raise ValueError("inference mode must be 2 for the five-crop pipeline")
    rgb = image_crop(frame, crop_idx)
    return Image.fromarray(rgb)


def encode_view(
    frames: list[np.ndarray],
    mode: int,
    crop_idx: int,
    clip_session: ort.InferenceSession,
    preprocess,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    feat_list: list[np.ndarray] = []
    model_seconds = 0.0
    input_name = clip_session.get_inputs()[0].name
    output_name = clip_session.get_outputs()[0].name
    fixed_batch_size = get_optional_onnx_batch_size(clip_session)
    loop_batch_size = fixed_batch_size if fixed_batch_size is not None else batch_size

    for start in range(0, len(frames), loop_batch_size):
        batch_frames = frames[start:start + loop_batch_size]
        batch_tensors = [
            preprocess(prepare_frame(frame, mode, crop_idx))
            for frame in batch_frames
        ]
        image_batch = torch.stack(batch_tensors, dim=0).numpy().astype(np.float32)
        valid_count = image_batch.shape[0]

        if fixed_batch_size is not None and valid_count < fixed_batch_size:
            padded_batch = np.zeros(
                (
                    fixed_batch_size,
                    image_batch.shape[1],
                    image_batch.shape[2],
                    image_batch.shape[3],
                ),
                dtype=np.float32,
            )
            padded_batch[:valid_count] = image_batch
            image_batch = padded_batch

        model_start = time.perf_counter()
        feat = clip_session.run([output_name], {input_name: image_batch})[0]
        model_seconds += time.perf_counter() - model_start
        feat_list.append(feat[:valid_count].astype(np.float32, copy=False))

    return np.concatenate(feat_list, axis=0).astype(np.float32), model_seconds


def extract_clip_features(
    video_path: Path,
    clip_session: ort.InferenceSession,
    preprocess,
    frame_stride: int,
    reference_fps: float,
    sample_position: str,
    mode: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    frames = sample_video_frames(video_path, frame_stride, reference_fps, sample_position)
    if mode != 2:
        raise ValueError("inference mode must be 2 for the five-crop pipeline")
    crop_indices = CROP_INDICES
    features = []
    model_seconds = 0.0
    for crop_idx in crop_indices:
        view_features, view_seconds = encode_view(
            frames,
            mode,
            crop_idx,
            clip_session,
            preprocess,
            batch_size,
        )
        features.append(view_features)
        model_seconds += view_seconds
    return np.stack(features, axis=0), model_seconds


def make_session(onnx_model: Path, provider: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider not available. Available: {available}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_model),
        sess_options=session_options,
        providers=providers,
    )
    print(f"ONNX Runtime providers: {session.get_providers()}")
    return session


def get_onnx_batch_size(session: ort.InferenceSession) -> int:
    shape = session.get_inputs()[0].shape
    batch_dim = shape[0]
    if isinstance(batch_dim, int) and batch_dim > 0:
        return batch_dim
    raise RuntimeError(
        "Cannot infer fixed ONNX batch size from model input. "
        "Export again with convert_2class_to_onnx.py."
    )


def get_optional_onnx_batch_size(session: ort.InferenceSession) -> int | None:
    shape = session.get_inputs()[0].shape
    batch_dim = shape[0]
    if isinstance(batch_dim, int) and batch_dim > 0:
        return batch_dim
    return None


def run_onnx_chunks(
    session: ort.InferenceSession,
    visual: np.ndarray,
    lengths: np.ndarray,
    onnx_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if visual.shape[0] != lengths.shape[0]:
        raise ValueError("visual and lengths batch size do not match")

    logits1_list: list[np.ndarray] = []
    logits2_list: list[np.ndarray] = []
    model_seconds = 0.0
    input_names = [input_meta.name for input_meta in session.get_inputs()]
    output_names = [output_meta.name for output_meta in session.get_outputs()]

    for start in range(0, visual.shape[0], onnx_batch_size):
        visual_batch = visual[start:start + onnx_batch_size]
        lengths_batch = lengths[start:start + onnx_batch_size]
        valid_count = visual_batch.shape[0]

        if valid_count < onnx_batch_size:
            padded_visual = np.zeros(
                (onnx_batch_size, visual.shape[1], visual.shape[2]),
                dtype=np.float32,
            )
            padded_lengths = np.ones((onnx_batch_size,), dtype=np.int64)
            padded_visual[:valid_count] = visual_batch
            padded_lengths[:valid_count] = lengths_batch
            visual_batch = padded_visual
            lengths_batch = padded_lengths

        ort_inputs = {
            input_names[0]: visual_batch.astype(np.float32, copy=False),
            input_names[1]: lengths_batch.astype(np.int64, copy=False),
        }
        model_start = time.perf_counter()
        logits1, logits2 = session.run(output_names, ort_inputs)
        model_seconds += time.perf_counter() - model_start
        logits1_list.append(logits1[:valid_count])
        logits2_list.append(logits2[:valid_count])

    return (
        np.concatenate(logits1_list, axis=0),
        np.concatenate(logits2_list, axis=0),
        model_seconds,
    )


def sigmoid_stable(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32, copy=False)
    positive = logits >= 0
    out = np.empty_like(logits, dtype=np.float32)
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    out[~positive] = exp_logits / (1.0 + exp_logits)
    return out


def infer_scores(
    session: ort.InferenceSession,
    clip_features: np.ndarray,
    visual_length: int,
    classes_num: int,
    spatial_top_k: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    crop_count, raw_length, _ = clip_features.shape
    split_crops = []
    chunk_lengths = None
    for crop_features in clip_features:
        chunks, lengths = split_feature(crop_features, visual_length)
        split_crops.append(chunks)
        if chunk_lengths is None:
            chunk_lengths = lengths
        elif not np.array_equal(chunk_lengths, lengths):
            raise RuntimeError("crop feature lengths do not match")

    visual = np.stack(split_crops).astype(np.float32)
    _, chunk_count, temporal_length, feature_dim = visual.shape
    visual = visual.reshape(
        crop_count * chunk_count,
        temporal_length,
        feature_dim,
    )
    lengths = np.tile(chunk_lengths, crop_count).astype(np.int64)

    onnx_batch_size = get_onnx_batch_size(session)
    logits1, logits2, model_seconds = run_onnx_chunks(session, visual, lengths, onnx_batch_size)

    logits1 = logits1.reshape(
        crop_count,
        chunk_count * temporal_length,
        1,
    )[:, :raw_length]
    logits2 = logits2.reshape(
        crop_count,
        chunk_count * temporal_length,
        classes_num,
    )[:, :raw_length]

    prob1 = sigmoid_stable(logits1.squeeze(-1))
    spatial_k = min(spatial_top_k, crop_count)
    prob1 = np.sort(prob1, axis=0)[-spatial_k:].mean(axis=0)

    shifted_logits2 = logits2 - logits2.max(axis=-1, keepdims=True)
    prob2_all = np.exp(shifted_logits2)
    prob2_all = prob2_all / prob2_all.sum(axis=-1, keepdims=True)
    prob2 = np.sort(prob2_all[..., 1], axis=0)[-spatial_k:].mean(axis=0)

    return prob1.astype(np.float32), prob2.astype(np.float32), model_seconds


def create_chart_base(
    scores1: np.ndarray,
    scores2: np.ndarray,
    width: int,
    chart_height: int,
) -> tuple[np.ndarray, list[int], int, int]:
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, chart_height / dpi), dpi=dpi)

    x = np.arange(len(scores1))
    ax.plot(x, scores1, color="#e74c3c", label="Score 1 (Coarse)", linewidth=2)
    ax.plot(x, scores2, color="#2ecc71", label="Score 2 (Fine)", linewidth=2)
    ax.fill_between(x, scores1, color="#e74c3c", alpha=0.15)
    ax.fill_between(x, scores2, color="#2ecc71", alpha=0.15)

    if len(scores1) <= 1:
        ax.set_xlim(0, 1)
    else:
        ax.set_xlim(0, len(scores1) - 1)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Anomaly Score", fontweight="bold", fontsize=10)
    ax.set_xlabel("Temporal Snippets (Time)", fontweight="bold", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    chart_img = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGB2BGR)

    x_pixels = []
    for i in range(len(scores1)):
        px, _ = ax.transData.transform((i, 0))
        x_pixels.append(int(px))

    _, py_bottom = ax.transData.transform((0, 0))
    _, py_top = ax.transData.transform((0, 1.0))
    y_bottom_cv = chart_img.shape[0] - int(py_bottom)
    y_top_cv = chart_img.shape[0] - int(py_top)

    plt.close(fig)

    h_orig, w_orig = chart_img.shape[:2]
    if w_orig != width or h_orig != chart_height:
        scale_x = width / w_orig
        scale_y = chart_height / h_orig
        chart_img = cv2.resize(chart_img, (width, chart_height))
        x_pixels = [int(px * scale_x) for px in x_pixels]
        y_top_cv = int(y_top_cv * scale_y)
        y_bottom_cv = int(y_bottom_cv * scale_y)

    return chart_img, x_pixels, y_top_cv, y_bottom_cv


def write_overlay_video(
    input_video: Path,
    output_video: Path,
    scores1: np.ndarray,
    scores2: np.ndarray,
    frame_stride: int,
    reference_fps: float,
) -> None:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if len(scores1) == 0 or len(scores2) == 0:
        raise RuntimeError("scores are empty")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for writing: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    chart_height = max(200, int(height * 0.3))
    base_chart, x_pixels, y_top_cv, y_bottom_cv = create_chart_base(
        scores1,
        scores2,
        width,
        chart_height,
    )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height + chart_height),
    )

    frame_id = 0
    max_idx = min(len(scores1), len(scores2)) - 1
    snippet_duration = frame_stride / reference_fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        clip_idx = int((frame_id / fps) / snippet_duration)
        if clip_idx > max_idx:
            clip_idx = max_idx

        cur_chart = base_chart.copy()
        line_x = x_pixels[clip_idx]
        cv2.line(cur_chart, (line_x, y_top_cv), (line_x, y_bottom_cv), (0, 0, 0), 2)

        score1 = max(0.0, min(1.0, float(scores1[clip_idx])))
        score2 = max(0.0, min(1.0, float(scores2[clip_idx])))
        cv2.putText(
            cur_chart,
            f"S1 (Coarse): {score1:.3f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 50, 200),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            cur_chart,
            f"S2 (Fine)  : {score2:.3f}",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 200, 50),
            2,
            cv2.LINE_AA,
        )

        writer.write(np.vstack((frame, cur_chart)))
        frame_id += 1

    writer.release()
    cap.release()


def iter_videos(root: Path) -> list[Path]:
    return sorted([
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    ])


def build_output_path(
    input_root: Path,
    video_path: Path,
    output_root: Path,
    suffix: str,
) -> Path:
    rel_path = video_path.relative_to(input_root)
    out_name = f"{rel_path.stem}{suffix}{rel_path.suffix}"
    return output_root / rel_path.parent / out_name


def collect_inputs(input_path: Path) -> tuple[Path, list[Path]]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"unsupported video extension: {input_path.suffix}")
        return input_path.parent, [input_path]
    if input_path.is_dir():
        videos = iter_videos(input_path)
        if not videos:
            raise RuntimeError(f"no videos found in folder: {input_path}")
        return input_path, videos
    raise FileNotFoundError(f"input not found: {input_path}")


def select_scores(scores1: np.ndarray, scores2: np.ndarray, score_source: str) -> np.ndarray:
    if score_source == "a1":
        return scores1
    if score_source == "a2":
        return scores2
    return (scores1 + scores2) / 2.0


def summarize_prediction(video_path: Path, scores1: np.ndarray, scores2: np.ndarray, score_source: str) -> dict:
    anomaly_scores = select_scores(scores1, scores2, score_source)
    max_score = float(anomaly_scores.max())
    min_score = float(anomaly_scores.min())
    prediction = "accident" if max_score > 0.5 else "normal"
    return {
        "video": str(video_path),
        "max_score": max_score,
        "min_score": min_score,
        "score_source": score_source,
        "prediction": prediction,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_root)
    vad_onnx_model = Path(args.vad_onnx_model) if args.vad_onnx_model else Path(args.onnx_model)
    clip_onnx_model = Path(args.clip_onnx_model)

    if not vad_onnx_model.exists():
        raise FileNotFoundError(f"VADCLIP ONNX model not found: {vad_onnx_model}")
    if not clip_onnx_model.exists():
        raise FileNotFoundError(f"CLIP ViT-B/16 ONNX model not found: {clip_onnx_model}")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if not 1 <= args.spatial_top_k <= len(CROP_INDICES):
        raise ValueError("--spatial-top-k must be between 1 and 5")

    input_root, videos = collect_inputs(input_path)
    suffix = args.suffix
    if suffix == "":
        suffix = "_onnx_5crop"

    print("[1/4] Loading ONNX CLIP ViT-B/16 image encoder...")
    preprocess = build_clip_preprocess(224)
    clip_session = make_session(clip_onnx_model, args.provider)
    clip_batch_size = get_optional_onnx_batch_size(clip_session)
    if clip_batch_size is None:
        print("CLIP ONNX batch size: dynamic")
    else:
        print(f"Fixed CLIP ONNX batch size: {clip_batch_size}")

    print("[2/4] Loading ONNX VAD model...")
    vad_session = make_session(vad_onnx_model, args.provider)
    print(f"Fixed VAD ONNX batch size: {get_onnx_batch_size(vad_session)}")

    rows = []
    mode_name = "five-crop"
    print(f"[3/4] Processing {len(videos)} video(s) in {mode_name} mode...")

    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path}")
        t0 = time.perf_counter()
        clip_features, vit_seconds = extract_clip_features(
            video_path,
            clip_session,
            preprocess,
            args.frame_stride,
            args.reference_fps,
            args.sample_position,
            args.inference_mode,
            args.clip_batch_size,
        )

        scores1, scores2, vad_seconds = infer_scores(
            vad_session,
            clip_features,
            args.visual_length,
            args.classes_num,
            args.spatial_top_k,
        )
        t2 = time.perf_counter()
        model_seconds = vit_seconds + vad_seconds
        end_to_end_seconds = t2 - t0

        row = summarize_prediction(video_path, scores1, scores2, args.score_source)
        row["model_seconds"] = model_seconds
        row["end_to_end_seconds"] = end_to_end_seconds
        rows.append(row)

        output_video = build_output_path(input_root, video_path, output_root, suffix)
        try:
            write_overlay_video(
                video_path,
                output_video,
                scores1,
                scores2,
                args.frame_stride,
                args.reference_fps,
            )
        except Exception as exc:
            print(f"Warning: cannot write overlay video for {video_path}: {exc}")

    print("[4/4] Saving summary...")
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / args.summary_name
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Output saved under: {output_root}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
