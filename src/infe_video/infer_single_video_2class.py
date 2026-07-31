import csv
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image
import time

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clip import clip
from crop import image_crop
from model import CLIPVAD
from ucf_staged_training_common import split_feature
from utils.tools import get_batch_mask, get_prompt_text

# =========================
# User parameters (edit here)
# =========================
INPUT_VIDEO_PATH = "C:/Users/ADMIN/Downloads/main_dataset/finetune_data/videos/val/Accident/000222.mp4"
OUTPUT_NAME = "test.mp4"
OUTPUT_DIR = str(PROJECT_DIR / "output" / "single_video_pytorch")
MODEL_PATH = "C:/Users/ADMIN/Desktop/python_code/python/thesis/main_model/VadCLIP-main/output/model_current.pth"
SUMMARY_NAME = "inference_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_STRIDE = 16
REFERENCE_FPS = 30.0
SAMPLE_POSITION = "first"  # first/middle/last
CLIP_BATCH_SIZE = 64

INFERENCE_MODE = 2
SPATIAL_TOP_K = 3
CROP_INDICES = (0, 1, 2, 3, 4)
USE_PADDING_MASK = False

# Model params (match training)
EMBED_DIM = 512
VISUAL_LENGTH = 256
VISUAL_WIDTH = 512
VISUAL_HEAD = 1
VISUAL_LAYERS = 2
ATTN_WINDOW = 8
PROMPT_PREFIX = 10
PROMPT_POSTFIX = 10
CLASSES_NUM = 2
LABEL_MAP = {
    "Normal": "normal",
    "Accident": "roadAccidents",
}
PROMPT_TEXT = get_prompt_text(LABEL_MAP)


def sync_if_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


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
            if previous_frame is not None and abs(previous_time - next_sample_time) < abs(frame_time - next_sample_time):
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
        raise ValueError("INFERENCE_MODE must be 2 for the five-crop pipeline")
    rgb = image_crop(frame, crop_idx)
    return Image.fromarray(rgb)


def encode_view(
    frames: list[np.ndarray],
    mode: int,
    crop_idx: int,
    clip_model,
    preprocess,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    feat_list: list[np.ndarray] = []
    model_seconds = 0.0

    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start:start + batch_size]
            batch_tensors = [
                preprocess(prepare_frame(frame, mode, crop_idx))
                for frame in batch_frames
            ]

            image_batch = torch.stack(batch_tensors, dim=0).to(device)
            sync_if_cuda(device)
            model_start = time.perf_counter()
            feat = clip_model.encode_image(image_batch)
            sync_if_cuda(device)
            model_seconds += time.perf_counter() - model_start
            feat_list.append(feat.float().cpu().numpy())

    return np.concatenate(feat_list, axis=0).astype(np.float32), model_seconds


def extract_clip_features(
    video_path: Path,
    clip_model,
    preprocess,
    device: str,
    frame_stride: int,
    reference_fps: float,
    sample_position: str,
    mode: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    frames = sample_video_frames(video_path, frame_stride, reference_fps, sample_position)
    if mode != 2:
        raise ValueError("INFERENCE_MODE must be 2 for the five-crop pipeline")
    crop_indices = CROP_INDICES
    features = []
    model_seconds = 0.0
    for crop_idx in crop_indices:
        view_features, view_seconds = encode_view(
            frames,
            mode,
            crop_idx,
            clip_model,
            preprocess,
            device,
            batch_size,
        )
        features.append(view_features)
        model_seconds += view_seconds
    return np.stack(features, axis=0), model_seconds


def load_model() -> CLIPVAD:
    model = CLIPVAD(
        CLASSES_NUM,
        EMBED_DIM,
        VISUAL_LENGTH,
        VISUAL_WIDTH,
        VISUAL_HEAD,
        VISUAL_LAYERS,
        ATTN_WINDOW,
        PROMPT_PREFIX,
        PROMPT_POSTFIX,
        DEVICE,
        USE_PADDING_MASK,
    )

    model_weights = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(model_weights, dict) and "model_state_dict" in model_weights:
        model_weights = model_weights["model_state_dict"]

    model.load_state_dict(model_weights)
    model.to(DEVICE)
    model.eval()
    return model


def infer_scores(
    model: CLIPVAD,
    clip_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    crop_count, raw_length, _ = clip_features.shape
    split_crops = []
    chunk_lengths = None
    for crop_features in clip_features:
        chunks, lengths = split_feature(crop_features, VISUAL_LENGTH)
        split_crops.append(chunks)
        if chunk_lengths is None:
            chunk_lengths = lengths
        elif not np.array_equal(chunk_lengths, lengths):
            raise RuntimeError("crop feature lengths do not match")

    visual = torch.from_numpy(np.stack(split_crops))
    _, chunk_count, temporal_length, feature_dim = visual.shape
    visual = visual.reshape(
        crop_count * chunk_count,
        temporal_length,
        feature_dim,
    ).to(DEVICE)
    lengths = torch.from_numpy(chunk_lengths).repeat(crop_count).to(DEVICE)
    padding_mask = None
    if USE_PADDING_MASK:
        padding_mask = get_batch_mask(
            lengths.detach().cpu(),
            VISUAL_LENGTH,
        ).to(DEVICE)

    sync_if_cuda(DEVICE)
    model_start = time.perf_counter()
    with torch.inference_mode():
        _, logits1, logits2 = model(
            visual,
            padding_mask,
            PROMPT_TEXT,
            lengths,
        )
    sync_if_cuda(DEVICE)
    model_seconds = time.perf_counter() - model_start

    logits1 = logits1.reshape(
        crop_count,
        chunk_count * temporal_length,
        1,
    )[:, :raw_length]
    logits2 = logits2.reshape(
        crop_count,
        chunk_count * temporal_length,
        CLASSES_NUM,
    )[:, :raw_length]

    prob1 = torch.sigmoid(logits1.squeeze(-1))
    spatial_k = min(SPATIAL_TOP_K, crop_count)
    prob1 = torch.topk(prob1, k=spatial_k, dim=0).values.mean(dim=0)

    prob2 = torch.softmax(logits2, dim=-1)[..., 1]
    prob2 = torch.topk(prob2, k=spatial_k, dim=0).values.mean(dim=0)

    return prob1.cpu().numpy(), prob2.cpu().numpy(), model_seconds


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

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
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

        score1 = float(scores1[clip_idx])
        score2 = float(scores2[clip_idx])
        score1 = max(0.0, min(1.0, score1))
        score2 = max(0.0, min(1.0, score2))

        cv2.putText(
            frame,
            f"A1: {score1:.4f}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"A2: {score2:.4f}",
            (8, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_id += 1

    writer.release()
    cap.release()


def summarize_prediction(video_path: Path, scores1: np.ndarray, scores2: np.ndarray) -> dict:
    anomaly_scores = (scores1 + scores2) / 2.0
    max_score = float(anomaly_scores.max())
    min_score = float(anomaly_scores.min())
    prediction = "accident" if max_score > 0.5 else "normal"
    return {
        "video": str(video_path),
        "max_score": max_score,
        "min_score": min_score,
        "prediction": prediction,
    }


def write_summary_csv(output_dir: Path, row: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_NAME
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return summary_path


def main() -> None:
    input_video = Path(INPUT_VIDEO_PATH)
    if not input_video.exists():
        raise FileNotFoundError(f"input video not found: {input_video}")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")
    if INFERENCE_MODE != 2:
        raise ValueError("INFERENCE_MODE must be 2 for the five-crop pipeline")
    if not 1 <= SPATIAL_TOP_K <= len(CROP_INDICES):
        raise ValueError("SPATIAL_TOP_K must be between 1 and 5")

    output_dir = Path(OUTPUT_DIR)
    output_name = OUTPUT_NAME
    if output_name == "":
        output_name = f"{input_video.stem}_my_model.mp4"
    output_video = output_dir / output_name

    print("[1/4] Loading CLIP ViT-B/16...")
    clip_model, preprocess = clip.load("ViT-B/16", device=DEVICE)
    clip_model.eval()

    start_time = time.time()
    mode_name = "five-crop"
    print(f"[2/4] Extracting CLIP features in {mode_name} mode...")
    clip_features, vit_seconds = extract_clip_features(
        input_video,
        clip_model,
        preprocess,
        DEVICE,
        FRAME_STRIDE,
        REFERENCE_FPS,
        SAMPLE_POSITION,
        INFERENCE_MODE,
        CLIP_BATCH_SIZE,
    )
    print(f"Extracted feature shape: {clip_features.shape}")

    print("[3/4] Loading VAD model and inferring scores...")
    model = load_model()
    scores1, scores2, vad_seconds = infer_scores(model, clip_features)
    score_ready_time = time.time()
    print(f"Predicted temporal clips: {scores1.shape[0]}")

    row = summarize_prediction(input_video, scores1, scores2)
    row["model_seconds"] = vit_seconds + vad_seconds
    row["end_to_end_seconds"] = score_ready_time - start_time
    summary_path = write_summary_csv(output_dir, row)

    print("[4/4] Writing overlay video...")
    write_overlay_video(input_video, output_video, scores1, scores2, FRAME_STRIDE, REFERENCE_FPS)

    end_time = time.time()
    print(f"Done. Output saved to: {output_video}")
    print(f"Summary CSV: {summary_path}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
