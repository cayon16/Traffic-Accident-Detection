from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from app.config import SYSTEM_DIR, Settings


CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
CROP_INDICES = (0, 1, 2, 3, 4)


class DetectorBackend(Protocol):
    def predict(self, frames: list[np.ndarray], timestamps: list[datetime]) -> dict:
        ...


class VADCLIPService:
    def __init__(self, settings: Settings):
        if settings.model_backend == "onnx":
            self._backend: DetectorBackend = ONNXVADCLIPBackend(settings)
        elif settings.model_backend == "pth":
            self._backend = PyTorchVADCLIPBackend(settings)
        elif settings.model_backend == "mock":
            self._backend = MockVADCLIPBackend()
        else:
            raise ValueError(f"unsupported model backend: {settings.model_backend}")

    def predict(self, frames: list[np.ndarray], timestamps: list[datetime]) -> dict:
        return self._backend.predict(frames, timestamps)


class MockVADCLIPBackend:
    def predict(self, frames: list[np.ndarray], timestamps: list[datetime]) -> dict:
        if not frames:
            return {
                "scores": [],
                "score_timestamps": [],
                "max_score": 0.0,
                "max_score_timestamp": "",
                "max_score_index": -1,
            }

        scores: list[float] = []
        previous_gray = None
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_gray is None:
                score = 0.05
            else:
                diff = cv2.absdiff(gray, previous_gray).mean()
                score = min(1.0, 0.05 + diff / 80.0)
            scores.append(float(score))
            previous_gray = gray

        return build_prediction(scores, timestamps)


class ONNXVADCLIPBackend:
    def __init__(self, settings: Settings):
        import onnxruntime as ort

        self.ort = ort
        self.clip_session = self._make_session(settings.clip_onnx_path, settings.onnx_provider)
        self.vad_session = self._make_session(settings.vad_onnx_path, settings.onnx_provider)
        self.score_source = settings.score_source
        self.inference_mode = settings.inference_mode
        self.spatial_top_k = settings.spatial_top_k
        self.visual_length = settings.visual_length
        self.classes_num = settings.classes_num
        self.clip_batch_size = settings.clip_batch_size

    def predict(self, frames: list[np.ndarray], timestamps: list[datetime]) -> dict:
        if not frames:
            return build_prediction([], [])

        clip_features = self._extract_clip_features(frames)
        score1, score2 = self._infer_scores(clip_features)
        if self.score_source == "a1":
            scores = score1
        elif self.score_source == "a2":
            scores = score2
        else:
            scores = (score1 + score2) / 2.0

        score_timestamps = align_timestamps(timestamps, len(scores))
        return build_prediction(scores.tolist(), score_timestamps)

    def _make_session(self, onnx_path: str, provider: str):
        path = Path(onnx_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        available = self.ort.get_available_providers()
        if provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(f"CUDAExecutionProvider not available: {available}")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")

        options = self.ort.SessionOptions()
        options.log_severity_level = 3
        return self.ort.InferenceSession(str(path), sess_options=options, providers=providers)

    def _extract_clip_features(self, frames: list[np.ndarray]) -> np.ndarray:
        if self.inference_mode != 2:
            raise ValueError("inference_mode must be 2 for the five-crop pipeline")
        crop_indices = CROP_INDICES
        features = []
        for crop_idx in crop_indices:
            features.append(self._encode_view(frames, crop_idx))
        return np.stack(features, axis=0).astype(np.float32)

    def _encode_view(self, frames: list[np.ndarray], crop_idx: int) -> np.ndarray:
        input_name = self.clip_session.get_inputs()[0].name
        output_name = self.clip_session.get_outputs()[0].name
        batch_size = self._optional_batch_size(self.clip_session) or self.clip_batch_size
        feature_batches = []

        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start:start + batch_size]
            batch = np.stack([
                preprocess_frame(frame, self.inference_mode, crop_idx)
                for frame in batch_frames
            ]).astype(np.float32)
            valid_count = batch.shape[0]
            fixed_batch_size = self._optional_batch_size(self.clip_session)
            if fixed_batch_size is not None and valid_count < fixed_batch_size:
                padded = np.zeros((fixed_batch_size, 3, 224, 224), dtype=np.float32)
                padded[:valid_count] = batch
                batch = padded

            features = self.clip_session.run([output_name], {input_name: batch})[0]
            feature_batches.append(features[:valid_count].astype(np.float32, copy=False))

        return np.concatenate(feature_batches, axis=0)

    def _infer_scores(self, clip_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        crop_count, raw_length, _ = clip_features.shape
        split_crops = []
        chunk_lengths = None
        for crop_features in clip_features:
            chunks, lengths = split_feature(crop_features, self.visual_length)
            split_crops.append(chunks)
            if chunk_lengths is None:
                chunk_lengths = lengths
            elif not np.array_equal(chunk_lengths, lengths):
                raise RuntimeError("crop feature lengths do not match")

        visual = np.stack(split_crops).astype(np.float32)
        _, chunk_count, temporal_length, feature_dim = visual.shape
        visual = visual.reshape(crop_count * chunk_count, temporal_length, feature_dim)
        lengths = np.tile(chunk_lengths, crop_count).astype(np.int64)

        logits1, logits2 = self._run_vad_chunks(visual, lengths)
        logits1 = logits1.reshape(crop_count, chunk_count * temporal_length, 1)[:, :raw_length]
        logits2 = logits2.reshape(
            crop_count,
            chunk_count * temporal_length,
            self.classes_num,
        )[:, :raw_length]

        spatial_k = min(self.spatial_top_k, crop_count)
        score1 = np.sort(sigmoid_stable(logits1.squeeze(-1)), axis=0)[-spatial_k:].mean(axis=0)
        prob2 = softmax(logits2, axis=-1)[..., 1]
        score2 = np.sort(prob2, axis=0)[-spatial_k:].mean(axis=0)
        return score1.astype(np.float32), score2.astype(np.float32)

    def _run_vad_chunks(self, visual: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        input_names = [item.name for item in self.vad_session.get_inputs()]
        output_names = [item.name for item in self.vad_session.get_outputs()]
        batch_size = self._fixed_batch_size(self.vad_session)
        logits1_list = []
        logits2_list = []

        for start in range(0, visual.shape[0], batch_size):
            visual_batch = visual[start:start + batch_size]
            lengths_batch = lengths[start:start + batch_size]
            valid_count = visual_batch.shape[0]
            if valid_count < batch_size:
                padded_visual = np.zeros((batch_size, visual.shape[1], visual.shape[2]), dtype=np.float32)
                padded_lengths = np.ones((batch_size,), dtype=np.int64)
                padded_visual[:valid_count] = visual_batch
                padded_lengths[:valid_count] = lengths_batch
                visual_batch = padded_visual
                lengths_batch = padded_lengths

            outputs = self.vad_session.run(
                output_names,
                {
                    input_names[0]: visual_batch.astype(np.float32, copy=False),
                    input_names[1]: lengths_batch.astype(np.int64, copy=False),
                },
            )
            logits1_list.append(outputs[0][:valid_count])
            logits2_list.append(outputs[1][:valid_count])

        return np.concatenate(logits1_list, axis=0), np.concatenate(logits2_list, axis=0)

    def _fixed_batch_size(self, session) -> int:
        batch_dim = session.get_inputs()[0].shape[0]
        if isinstance(batch_dim, int) and batch_dim > 0:
            return batch_dim
        raise RuntimeError("VAD ONNX model must use a fixed batch size")

    def _optional_batch_size(self, session) -> int | None:
        batch_dim = session.get_inputs()[0].shape[0]
        if isinstance(batch_dim, int) and batch_dim > 0:
            return batch_dim
        return None


class PyTorchVADCLIPBackend:
    def __init__(self, settings: Settings):
        import torch

        src_dir = SYSTEM_DIR.parent / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from clip import clip
        from model import CLIPVAD
        from utils.tools import get_prompt_text

        model_path = Path(settings.pth_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"PyTorch checkpoint not found: {model_path}")

        if settings.pth_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = settings.pth_device

        self.torch = torch
        self.score_source = settings.score_source
        self.inference_mode = settings.inference_mode
        self.spatial_top_k = settings.spatial_top_k
        self.visual_length = settings.visual_length
        self.classes_num = settings.classes_num
        self.clip_batch_size = settings.clip_batch_size
        self.prompt_text = get_prompt_text({"Normal": "normal", "Accident": "roadAccidents"})

        download_root = settings.clip_download_root or None
        self.clip_model, self.preprocess = clip.load(
            "ViT-B/16",
            self.device,
            download_root=download_root,
        )
        self.clip_model.eval()

        self.model = CLIPVAD(
            self.classes_num,
            512,
            self.visual_length,
            512,
            1,
            2,
            8,
            10,
            10,
            self.device,
            False,
        )
        checkpoint = torch.load(str(model_path), map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, frames: list[np.ndarray], timestamps: list[datetime]) -> dict:
        if not frames:
            return build_prediction([], [])

        clip_features = self._extract_clip_features(frames)
        score1, score2 = self._infer_scores(clip_features)
        if self.score_source == "a1":
            scores = score1
        elif self.score_source == "a2":
            scores = score2
        else:
            scores = (score1 + score2) / 2.0

        score_timestamps = align_timestamps(timestamps, len(scores))
        return build_prediction(scores.tolist(), score_timestamps)

    def _extract_clip_features(self, frames: list[np.ndarray]) -> np.ndarray:
        if self.inference_mode != 2:
            raise ValueError("inference_mode must be 2 for the five-crop pipeline")
        features = [
            self._encode_view(frames, crop_idx)
            for crop_idx in CROP_INDICES
        ]
        return np.stack(features, axis=0).astype(np.float32)

    def _encode_view(self, frames: list[np.ndarray], crop_idx: int) -> np.ndarray:
        feature_batches = []
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, len(frames), self.clip_batch_size):
                batch_frames = frames[start:start + self.clip_batch_size]
                image_batch = torch.stack(
                    [
                        self.preprocess(Image.fromarray(crop_frame(frame, crop_idx)))
                        for frame in batch_frames
                    ],
                    dim=0,
                ).to(self.device)
                features = self.clip_model.encode_image(image_batch)
                feature_batches.append(features.float().cpu().numpy())
        return np.concatenate(feature_batches, axis=0).astype(np.float32)

    def _infer_scores(self, clip_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        crop_count, raw_length, _ = clip_features.shape
        split_crops = []
        chunk_lengths = None
        for crop_features in clip_features:
            chunks, lengths = split_feature(crop_features, self.visual_length)
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
        ).to(self.device)
        lengths = torch.from_numpy(chunk_lengths).repeat(crop_count).to(self.device)

        with torch.inference_mode():
            _, logits1, logits2 = self.model(
                visual,
                None,
                self.prompt_text,
                lengths,
            )

        logits1 = logits1.reshape(
            crop_count,
            chunk_count * temporal_length,
            1,
        )[:, :raw_length]
        logits2 = logits2.reshape(
            crop_count,
            chunk_count * temporal_length,
            self.classes_num,
        )[:, :raw_length]

        spatial_k = min(self.spatial_top_k, crop_count)
        score1 = torch.topk(
            torch.sigmoid(logits1.squeeze(-1)),
            k=spatial_k,
            dim=0,
        ).values.mean(dim=0)
        score2 = torch.topk(
            torch.softmax(logits2, dim=-1)[..., 1],
            k=spatial_k,
            dim=0,
        ).values.mean(dim=0)
        return score1.cpu().numpy().astype(np.float32), score2.cpu().numpy().astype(np.float32)


def preprocess_frame(frame_bgr: np.ndarray, mode: int, crop_idx: int) -> np.ndarray:
    if mode != 2:
        raise ValueError("inference_mode must be 2 for the five-crop pipeline")
    rgb = crop_frame(frame_bgr, crop_idx)

    image = rgb.astype(np.float32) / 255.0
    image = (image - CLIP_MEAN) / CLIP_STD
    return np.transpose(image, (2, 0, 1))


def crop_frame(frame_bgr: np.ndarray, crop_idx: int) -> np.ndarray:
    image = cv2.resize(frame_bgr, (340, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop_idx == 0:
        return image[16:240, 58:282, :]
    if crop_idx == 1:
        return image[:224, :224, :]
    if crop_idx == 2:
        return image[:224, -224:, :]
    if crop_idx == 3:
        return image[-224:, :224, :]
    if crop_idx == 4:
        return image[-224:, -224:, :]
    raise ValueError("crop_idx must be between 0 and 4")


def split_feature(feature: np.ndarray, visual_length: int) -> tuple[np.ndarray, np.ndarray]:
    raw_length = int(feature.shape[0])
    chunk_count = max(1, int(np.ceil(raw_length / visual_length)))
    chunks = np.zeros((chunk_count, visual_length, feature.shape[1]), dtype=np.float32)
    lengths = np.zeros(chunk_count, dtype=np.int64)
    for chunk_index in range(chunk_count):
        start = chunk_index * visual_length
        end = min(start + visual_length, raw_length)
        chunk_length = end - start
        chunks[chunk_index, :chunk_length] = feature[start:end]
        lengths[chunk_index] = chunk_length
    return chunks, lengths


def sigmoid_stable(logits: np.ndarray) -> np.ndarray:
    positive = logits >= 0
    out = np.empty_like(logits, dtype=np.float32)
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    out[~positive] = exp_logits / (1.0 + exp_logits)
    return out


def softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=axis, keepdims=True)


def align_timestamps(timestamps: list[datetime], score_count: int) -> list[datetime]:
    if score_count <= 0:
        return []
    if len(timestamps) == score_count:
        return timestamps
    if len(timestamps) < score_count:
        return timestamps + [timestamps[-1]] * (score_count - len(timestamps))

    indices = np.linspace(0, len(timestamps) - 1, score_count).round().astype(int)
    return [timestamps[int(index)] for index in indices]


def build_prediction(scores: list[float], timestamps: list[datetime]) -> dict:
    score_timestamps = [
        ts.replace(microsecond=0).isoformat() if isinstance(ts, datetime) else str(ts)
        for ts in timestamps
    ]
    if not scores:
        return {
            "scores": [],
            "score_timestamps": [],
            "max_score": 0.0,
            "max_score_timestamp": "",
            "max_score_index": -1,
        }
    max_index = max(range(len(scores)), key=lambda idx: scores[idx])
    return {
        "scores": [float(score) for score in scores],
        "score_timestamps": score_timestamps,
        "max_score": float(scores[max_index]),
        "max_score_timestamp": score_timestamps[max_index],
        "max_score_index": max_index,
    }
