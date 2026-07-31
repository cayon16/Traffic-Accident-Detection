"""Shared five-crop dataset, model-forward, and evaluation helpers."""

"""Shared data, evaluation, and utility helpers for staged training."""

from pathlib import Path
import math
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import CLIPVAD
from utils.tools import get_batch_mask, get_prompt_text, process_feat


LABEL_MAP = {
    "Normal": "normal",
    "Accident": "roadAccidents",
}
KNOWN_FILE_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".npy",
}


def canonical_video_name(value) -> str:
    name = Path(str(value).strip().replace("\\", "/")).name
    if Path(name).suffix.casefold() in KNOWN_FILE_EXTENSIONS:
        name = Path(name).stem
    if name.isdigit():
        name = str(int(name))
    return name.casefold()


def parse_feature_name(value) -> tuple[str, int]:
    stem = Path(str(value).strip().replace("\\", "/")).stem
    match = re.fullmatch(r"(.+)__(\d+)", stem)
    if match is None:
        raise ValueError(f"feature name has no crop suffix: {value}")
    return canonical_video_name(match.group(1)), int(match.group(2))


def load_timestamp_map(excel_path: str) -> dict[str, float]:
    table = pd.read_excel(excel_path)
    if table.shape[1] != 2:
        raise ValueError(
            f"timestamp Excel must contain exactly 2 columns, "
            f"found {table.shape[1]}"
        )

    timestamp_map: dict[str, float] = {}
    for video_value, time_value in table.iloc[:, :2].itertuples(
        index=False,
        name=None,
    ):
        if pd.isna(video_value) or pd.isna(time_value):
            raise ValueError(
                "timestamp Excel contains an empty video name or timestamp"
            )

        video_key = canonical_video_name(video_value)
        timestamp_seconds = float(time_value)
        if not video_key:
            raise ValueError("timestamp Excel contains an empty video name")
        if not np.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError(f"invalid timestamp for video: {video_value}")
        if video_key in timestamp_map and not np.isclose(
            timestamp_map[video_key],
            timestamp_seconds,
        ):
            raise ValueError(f"conflicting timestamps for video: {video_value}")
        timestamp_map[video_key] = timestamp_seconds

    if not timestamp_map:
        raise ValueError("timestamp Excel contains no annotations")
    return timestamp_map


def build_video_records(
    csv_path: str,
    crop_count: int,
    label_filter: str | None = None,
) -> list[dict]:
    table = pd.read_csv(csv_path)
    if not {"path", "label"}.issubset(table.columns):
        raise ValueError(f"CSV must contain path,label columns: {csv_path}")

    table = table.copy()
    parsed = table["path"].map(parse_feature_name)
    table["video_key"] = parsed.map(lambda item: item[0])
    table["crop_index"] = parsed.map(lambda item: item[1])
    table["normalized_label"] = (
        table["label"].astype(str).str.strip().str.casefold()
    )

    valid_labels = {label.casefold() for label in LABEL_MAP}
    invalid_labels = sorted(set(table["normalized_label"]) - valid_labels)
    if invalid_labels:
        raise ValueError(f"unsupported labels in {csv_path}: {invalid_labels}")

    if label_filter is not None:
        table = table.loc[
            table["normalized_label"] == label_filter.casefold()
        ]

    expected_crops = list(range(crop_count))
    records: list[dict] = []
    for video_key, group in table.groupby("video_key", sort=False):
        labels = group["normalized_label"].unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"conflicting labels for video: {video_key}")

        crop_indices = sorted(group["crop_index"].tolist())
        if crop_indices != expected_crops:
            raise ValueError(
                f"video {video_key} must contain crops {expected_crops}, "
                f"found {crop_indices}"
            )

        ordered = group.sort_values("crop_index")
        paths = [Path(path) for path in ordered["path"]]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])

        records.append(
            {
                "video_key": video_key,
                "label": str(ordered.iloc[0]["label"]).strip(),
                "paths": paths,
            }
        )

    if not records:
        suffix = f" for label {label_filter}" if label_filter else ""
        raise ValueError(f"no grouped videos found in {csv_path}{suffix}")
    return records


def split_feature(
    feature: np.ndarray,
    visual_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_length = int(feature.shape[0])
    chunk_count = max(1, math.ceil(raw_length / visual_length))
    chunks = np.zeros(
        (chunk_count, visual_length, feature.shape[1]),
        dtype=np.float32,
    )
    lengths = np.zeros(chunk_count, dtype=np.int64)

    for chunk_index in range(chunk_count):
        start = chunk_index * visual_length
        end = min(start + visual_length, raw_length)
        chunk_length = end - start
        chunks[chunk_index, :chunk_length] = feature[start:end]
        lengths[chunk_index] = chunk_length
    return chunks, lengths


class FiveCropVideoDataset(Dataset):
    def __init__(
        self,
        visual_length: int,
        csv_path: str,
        crop_count: int,
        label_filter: str | None = None,
        test_mode: bool = False,
        timestamp_map: dict[str, float] | None = None,
        base_snippet_seconds: float | None = None,
        timestamp_optional: bool = False,
    ):
        records = build_video_records(csv_path, crop_count, label_filter)
        if timestamp_map is not None:
            available_keys = {record["video_key"] for record in records}
            unmatched = sorted(set(timestamp_map) - available_keys)
            if unmatched:
                preview = ", ".join(unmatched[:10])
                raise ValueError(
                    f"{len(unmatched)} timestamp videos were not found in "
                    f"accident training features: {preview}"
                )
            if not timestamp_optional:
                records = [
                    record
                    for record in records
                    if record["video_key"] in timestamp_map
                ]

        if not records:
            raise ValueError("no videos remain after timestamp filtering")

        self.records = records
        self.visual_length = visual_length
        self.crop_count = crop_count
        self.test_mode = test_mode
        self.timestamp_map = timestamp_map
        self.base_snippet_seconds = base_snippet_seconds
        self.timestamp_optional = timestamp_optional

        print(
            f"Loaded {len(records)} grouped videos"
            f"{' for testing' if test_mode else ''}"
            f"{' with optional timestamps' if timestamp_optional else ''}"
            f"{' with timestamps' if timestamp_map is not None and not timestamp_optional else ''}"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        raw_features = [
            np.load(path).astype(np.float32, copy=False)
            for path in record["paths"]
        ]
        raw_lengths = [int(feature.shape[0]) for feature in raw_features]
        if min(raw_lengths) <= 0:
            raise ValueError(f"empty feature for video: {record['video_key']}")
        if len(set(raw_lengths)) != 1:
            raise ValueError(
                f"crop lengths differ for video {record['video_key']}: "
                f"{raw_lengths}"
            )
        if any(
            feature.ndim != 2 or feature.shape[1] != 512
            for feature in raw_features
        ):
            raise ValueError(
                f"invalid feature shape for video: {record['video_key']}"
            )

        raw_length = raw_lengths[0]
        if self.test_mode:
            split_crops = []
            chunk_lengths = None
            for feature in raw_features:
                chunks, lengths = split_feature(feature, self.visual_length)
                split_crops.append(chunks)
                if chunk_lengths is None:
                    chunk_lengths = lengths
                elif not np.array_equal(chunk_lengths, lengths):
                    raise ValueError(
                        f"crop chunks differ for video: {record['video_key']}"
                    )
            return (
                torch.from_numpy(np.stack(split_crops)),
                record["label"],
                torch.from_numpy(chunk_lengths),
                raw_length,
            )

        processed_crops = []
        feature_length = None
        for feature in raw_features:
            processed, length = process_feat(feature, self.visual_length)
            processed_crops.append(processed.astype(np.float32, copy=False))
            if feature_length is None:
                feature_length = int(length)
            elif feature_length != int(length):
                raise ValueError(
                    f"processed crop lengths differ: {record['video_key']}"
                )

        result = (
            torch.from_numpy(np.stack(processed_crops)),
            record["label"],
            feature_length,
        )
        if self.timestamp_map is None:
            return result

        timestamp = self.timestamp_map.get(
            record["video_key"],
            float("nan"),
        )
        video_duration = raw_length * self.base_snippet_seconds
        if np.isfinite(timestamp) and timestamp >= video_duration:
            raise ValueError(
                f"timestamp {timestamp:.3f}s is outside feature duration "
                f"{video_duration:.3f}s for video {record['video_key']}"
            )
        seconds_per_processed_snippet = video_duration / feature_length
        return result + (timestamp, seconds_per_processed_snippet)


def get_top_k_values(lengths, divisor, minimum, maximum):
    divisor = max(1, divisor)
    minimum = max(1, minimum)
    maximum = max(minimum, maximum)
    k_values = lengths // divisor + 1
    k_values = torch.clamp(k_values, min=minimum, max=maximum)
    return torch.minimum(k_values, lengths)


def labels_to_indices(labels, device):
    return torch.tensor(
        [0 if str(label).strip().casefold() == "normal" else 1 for label in labels],
        dtype=torch.long,
        device=device,
    )


def forward_five_crop(
    model,
    features,
    lengths,
    prompt_text,
    use_padding_mask,
    device,
):
    batch_size, crop_count, temporal_length, feature_dim = features.shape
    flat_features = features.reshape(
        batch_size * crop_count,
        temporal_length,
        feature_dim,
    ).to(device)
    flat_lengths = lengths.to(device).repeat_interleave(crop_count)

    padding_mask = None
    if use_padding_mask:
        padding_mask = get_batch_mask(
            flat_lengths.detach().cpu(),
            temporal_length,
        ).to(device)

    text_features, logits1, logits2 = model(
        flat_features,
        padding_mask,
        prompt_text,
        flat_lengths,
    )
    logits1 = logits1.reshape(
        batch_size,
        crop_count,
        temporal_length,
        1,
    )
    logits2 = logits2.reshape(
        batch_size,
        crop_count,
        temporal_length,
        logits2.shape[-1],
    )
    return text_features, logits1, logits2


def aggregate_spatial_scores(logits1, logits2, spatial_top_k):
    scores_a1 = torch.sigmoid(logits1.squeeze(-1))
    scores_a2 = torch.softmax(logits2, dim=-1)[..., 1]
    scores_a1 = torch.topk(
        scores_a1,
        k=spatial_top_k,
        dim=1,
    ).values.mean(dim=1)
    scores_a2 = torch.topk(
        scores_a2,
        k=spatial_top_k,
        dim=1,
    ).values.mean(dim=1)
    return scores_a1, scores_a2


def aggregate_spatial_a2_logits(logits2, spatial_top_k):
    anomaly_margin = logits2[..., 1] - logits2[..., 0]
    crop_indices = torch.topk(
        anomaly_margin,
        k=spatial_top_k,
        dim=1,
    ).indices
    gather_indices = crop_indices.unsqueeze(-1).expand(
        -1,
        -1,
        -1,
        logits2.shape[-1],
    )
    selected_logits = torch.gather(logits2, 1, gather_indices)
    return selected_logits.mean(dim=1)


def mil_a1_loss(scores, labels, lengths, k_values):
    video_scores = []
    for index in range(scores.shape[0]):
        length = int(lengths[index])
        k_value = int(k_values[index])
        top_scores = torch.topk(
            scores[index, :length],
            k=k_value,
        ).values
        video_scores.append(top_scores.mean())
    return F.binary_cross_entropy(
        torch.stack(video_scores),
        labels.to(scores.dtype),
    )


def mil_a2_loss(logits, labels, lengths, k_values):
    video_logits = []
    for index in range(logits.shape[0]):
        length = int(lengths[index])
        k_value = int(k_values[index])
        valid_logits = logits[index, :length]
        anomaly_margin = valid_logits[:, 1] - valid_logits[:, 0]
        temporal_indices = torch.topk(
            anomaly_margin,
            k=k_value,
        ).indices
        video_logits.append(
            valid_logits.index_select(0, temporal_indices).mean(dim=0)
        )
    return F.cross_entropy(torch.stack(video_logits), labels)


def compute_mil_losses(
    logits1,
    logits2,
    labels,
    lengths,
    k_values,
    spatial_top_k,
):
    scores_a1, _ = aggregate_spatial_scores(
        logits1,
        logits2,
        spatial_top_k,
    )
    spatial_logits_a2 = aggregate_spatial_a2_logits(
        logits2,
        spatial_top_k,
    )
    loss1 = mil_a1_loss(
        scores_a1,
        labels,
        lengths,
        k_values,
    )
    loss2 = mil_a2_loss(
        spatial_logits_a2,
        labels,
        lengths,
        k_values,
    )
    return loss1, loss2


def text_separation_loss(text_features, weight):
    normal_feature = text_features[0] / text_features[0].norm().clamp_min(1e-12)
    losses = []
    for index in range(1, text_features.shape[0]):
        accident_feature = (
            text_features[index]
            / text_features[index].norm().clamp_min(1e-12)
        )
        losses.append(torch.abs(normal_feature @ accident_feature))
    if not losses:
        return text_features.sum() * 0.0
    return torch.stack(losses).mean() * weight


def region_binary_loss(scores, indices, target):
    if indices.numel() == 0:
        return None
    selected_scores = scores.index_select(0, indices)
    targets = torch.full_like(selected_scores, target)
    return F.binary_cross_entropy(selected_scores, targets)


def temporal_supervision(
    scores_a1,
    scores_a2,
    normal_batch_size,
    accident_lengths,
    timestamps,
    seconds_per_snippet,
    accident_k_values,
    args,
):
    temporal_a1 = []
    temporal_a2 = []
    gap_a1 = []
    gap_a2 = []

    for local_index in range(accident_lengths.shape[0]):
        sample_index = normal_batch_size + local_index
        length = int(accident_lengths[local_index])
        timestamp = float(timestamps[local_index])
        snippet_seconds = float(seconds_per_snippet[local_index])
        k_value = int(accident_k_values[local_index])

        pre_end = math.floor(
            (timestamp - args.pre_normal_buffer_seconds) / snippet_seconds
        )
        accident_start = math.floor(
            (timestamp + args.accident_window_start_seconds) / snippet_seconds
        )
        accident_end = math.ceil(
            (timestamp + args.accident_window_end_seconds) / snippet_seconds
        )
        post_start = math.ceil(
            (timestamp + args.post_normal_start_seconds) / snippet_seconds
        )

        pre_end = min(max(pre_end, 0), length)
        accident_start = min(max(accident_start, 0), length)
        accident_end = min(max(accident_end, accident_start + 1), length)
        post_start = min(max(post_start, accident_end), length)

        device = scores_a1.device
        pre_indices = torch.arange(0, pre_end, device=device)
        accident_indices = torch.arange(
            accident_start,
            accident_end,
            device=device,
        )
        post_indices = torch.arange(post_start, length, device=device)

        for branch_scores, temporal_losses, gap_losses in (
            (scores_a1, temporal_a1, gap_a1),
            (scores_a2, temporal_a2, gap_a2),
        ):
            scores = branch_scores[sample_index, :length]
            normal_losses = []

            pre_loss = region_binary_loss(
                scores,
                pre_indices,
                args.normal_target,
            )
            if pre_loss is not None:
                normal_losses.append((pre_loss, 1.0))

            post_loss = region_binary_loss(
                scores,
                post_indices,
                args.normal_target,
            )
            if post_loss is not None:
                normal_losses.append(
                    (post_loss, args.post_normal_weight)
                )

            if normal_losses:
                total_weight = sum(weight for _, weight in normal_losses)
                normal_loss = sum(
                    loss * weight for loss, weight in normal_losses
                ) / total_weight
            else:
                normal_loss = scores.sum() * 0.0

            accident_loss = region_binary_loss(
                scores,
                accident_indices,
                args.accident_target,
            )
            if accident_loss is None:
                accident_loss = scores.sum() * 0.0
            temporal_losses.append(normal_loss + accident_loss)

            normal_indices = torch.cat([pre_indices, post_indices])
            if normal_indices.numel() and accident_indices.numel():
                gap_k = min(
                    k_value,
                    int(normal_indices.numel()),
                    int(accident_indices.numel()),
                )
                hard_normal = torch.topk(
                    scores.index_select(0, normal_indices),
                    k=gap_k,
                ).values.mean()
                accident_score = torch.topk(
                    scores.index_select(0, accident_indices),
                    k=gap_k,
                ).values.mean()
                gap_losses.append(
                    F.relu(
                        args.gap_margin - accident_score + hard_normal
                    )
                )
            else:
                gap_losses.append(scores.sum() * 0.0)

    zero_a1 = scores_a1.sum() * 0.0
    zero_a2 = scores_a2.sum() * 0.0
    return (
        torch.stack(temporal_a1).mean() if temporal_a1 else zero_a1,
        torch.stack(temporal_a2).mean() if temporal_a2 else zero_a2,
        torch.stack(gap_a1).mean() if gap_a1 else zero_a1,
        torch.stack(gap_a2).mean() if gap_a2 else zero_a2,
    )


def safe_metric(metric_function, targets, predictions):
    if len(np.unique(targets)) < 2:
        return None
    return metric_function(targets, predictions)


def cohen_d_score_gap(targets, predictions):
    targets = np.asarray(targets).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    accident_scores = predictions[targets > 0.5]
    normal_scores = predictions[targets <= 0.5]
    if accident_scores.size < 2 or normal_scores.size < 2:
        return None

    accident_var = accident_scores.var(ddof=1)
    normal_var = normal_scores.var(ddof=1)
    pooled_var = (
        (accident_scores.size - 1) * accident_var
        + (normal_scores.size - 1) * normal_var
    ) / (accident_scores.size + normal_scores.size - 2)
    if pooled_var <= 1e-12:
        return None
    return float(
        (accident_scores.mean() - normal_scores.mean())
        / np.sqrt(pooled_var)
    )


def format_metric(value):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.6f}"


def print_train_summary(stage, epoch, total_epochs, metrics, learning_rate):
    labels = {
        "loss1": "MIL loss A1",
        "loss2": "MIL loss A2",
        "loss3": "Text loss",
        "phase1_loss": "MIL + text loss",
        "temp_a1": "Temporal loss A1",
        "temp_a2": "Temporal loss A2",
        "gap_a1": "Gap loss A1 (raw)",
        "gap_a2": "Gap loss A2 (raw)",
        "gap_weight": "Gap weight",
        "total": "Total loss",
    }
    print("\n" + "=" * 62)
    print(f"{stage.upper()} | EPOCH {epoch}/{total_epochs}")
    print("=" * 62)
    print("Training losses")
    print("-" * 62)
    for name, value in metrics.items():
        label = labels.get(name, name.replace("_", " ").title())
        print(f"{label:<25}: {float(value):.6f}")
    print(f"{'Learning rate':<25}: {learning_rate:.8g}")


def print_validation_summary(metrics, checkpoint_value=None):
    print("\nValidation metrics")
    print("-" * 62)
    print(f"{'Metric':<22}{'A1':>12}{'A2':>12}{'Fused':>12}")
    print("-" * 62)
    rows = (
        ("Overall AUC", "auc1", "auc2", "aucf"),
        ("Overall AP", "ap1", "ap2", "apf"),
        ("Annotated AUC", "anno_auc1", "anno_auc2", "anno_aucf"),
        ("Annotated AP", "anno_ap1", "anno_ap2", "anno_apf"),
        ("FAR", "far1", "far2", "farf"),
    )
    for label, key_a1, key_a2, key_fused in rows:
        print(
            f"{label:<22}"
            f"{format_metric(metrics[key_a1]):>12}"
            f"{format_metric(metrics[key_a2]):>12}"
            f"{format_metric(metrics[key_fused]):>12}"
        )
    if checkpoint_value is not None:
        print("-" * 62)
        print(
            f"{'Checkpoint metric':<25}: "
            f"{format_metric(checkpoint_value)}"
        )
    print("=" * 62)


def evaluate_five_crop(
    model,
    test_loader,
    prompt_text,
    gt,
    args,
    device,
):
    model.eval()
    all_a1 = []
    all_a2 = []
    accident_gt = []
    accident_a1 = []
    accident_a2 = []
    normal_false_a1 = 0
    normal_false_a2 = 0
    normal_false_fused = 0
    normal_total = 0
    gt_offset = 0

    with torch.no_grad():
        for features, labels, chunk_lengths, raw_lengths in test_loader:
            features = features.squeeze(0)
            chunk_lengths = chunk_lengths.squeeze(0)
            raw_length = int(raw_lengths[0])
            crop_count, chunk_count, temporal_length, feature_dim = (
                features.shape
            )

            flat_features = features.reshape(
                crop_count * chunk_count,
                temporal_length,
                feature_dim,
            ).to(device)
            flat_lengths = chunk_lengths.repeat(crop_count).to(device)
            padding_mask = None
            if args.use_padding_mask:
                padding_mask = get_batch_mask(
                    flat_lengths.detach().cpu(),
                    temporal_length,
                ).to(device)

            _, logits1, logits2 = model(
                flat_features,
                padding_mask,
                prompt_text,
                flat_lengths,
            )
            logits1 = logits1.reshape(
                crop_count,
                chunk_count * temporal_length,
                1,
            )[:, :raw_length]
            logits2 = logits2.reshape(
                crop_count,
                chunk_count * temporal_length,
                2,
            )[:, :raw_length]

            spatial_k = min(args.spatial_top_k, crop_count)

            prob_a1 = torch.sigmoid(logits1.squeeze(-1))
            prob_a1 = torch.topk(
                prob_a1,
                k=spatial_k,
                dim=0,
            ).values.mean(dim=0)

            prob_a2 = torch.softmax(logits2, dim=-1)[..., 1]
            prob_a2 = torch.topk(
                prob_a2,
                k=spatial_k,
                dim=0,
            ).values.mean(dim=0)

            pred_a1 = np.repeat(
                prob_a1.cpu().numpy(),
                args.frames_per_snippet,
            )
            pred_a2 = np.repeat(
                prob_a2.cpu().numpy(),
                args.frames_per_snippet,
            )
            pred_fused = (pred_a1 + pred_a2) / 2.0
            is_normal = str(labels[0]).strip().casefold() == "normal"

            gt_end = gt_offset + len(pred_a1)
            if gt_end > len(gt):
                raise ValueError(
                    "ground truth is shorter than five-crop validation "
                    "predictions; regenerate gt_ucf.npy for the new val set"
                )
            gt_slice = gt[gt_offset:gt_end]
            gt_offset = gt_end

            all_a1.append(pred_a1)
            all_a2.append(pred_a2)
            if is_normal:
                normal_total += len(pred_a1)
                normal_false_a1 += int((pred_a1 > 0.5).sum())
                normal_false_a2 += int((pred_a2 > 0.5).sum())
                normal_false_fused += int((pred_fused > 0.5).sum())
            else:
                accident_gt.append(gt_slice)
                accident_a1.append(pred_a1)
                accident_a2.append(pred_a2)

    if gt_offset != len(gt):
        raise ValueError(
            f"ground truth has {len(gt) - gt_offset} extra frames; "
            "regenerate gt_ucf.npy in the exact five-crop CSV video order"
        )

    pred_a1 = np.concatenate(all_a1)
    pred_a2 = np.concatenate(all_a2)
    pred_fused = (pred_a1 + pred_a2) / 2.0
    metrics = {
        "auc1": safe_metric(roc_auc_score, gt, pred_a1),
        "ap1": average_precision_score(gt, pred_a1),
        "auc2": safe_metric(roc_auc_score, gt, pred_a2),
        "ap2": average_precision_score(gt, pred_a2),
        "aucf": safe_metric(roc_auc_score, gt, pred_fused),
        "apf": average_precision_score(gt, pred_fused),
        "far1": normal_false_a1 / normal_total if normal_total else None,
        "far2": normal_false_a2 / normal_total if normal_total else None,
        "farf": normal_false_fused / normal_total if normal_total else None,
        "anno_auc1": None,
        "anno_ap1": None,
        "anno_auc2": None,
        "anno_ap2": None,
        "anno_aucf": None,
        "anno_apf": None,
        "anno_cohen_d1": None,
        "anno_cohen_d2": None,
    }

    if accident_gt:
        anno_gt = np.concatenate(accident_gt)
        anno_a1 = np.concatenate(accident_a1)
        anno_a2 = np.concatenate(accident_a2)
        anno_fused = (anno_a1 + anno_a2) / 2.0
        metrics.update(
            {
                "anno_auc1": safe_metric(
                    roc_auc_score,
                    anno_gt,
                    anno_a1,
                ),
                "anno_ap1": average_precision_score(anno_gt, anno_a1),
                "anno_auc2": safe_metric(
                    roc_auc_score,
                    anno_gt,
                    anno_a2,
                ),
                "anno_ap2": average_precision_score(anno_gt, anno_a2),
                "anno_aucf": safe_metric(
                    roc_auc_score,
                    anno_gt,
                    anno_fused,
                ),
                "anno_apf": average_precision_score(anno_gt, anno_fused),
                "anno_cohen_d1": cohen_d_score_gap(anno_gt, anno_a1),
                "anno_cohen_d2": cohen_d_score_gap(anno_gt, anno_a2),
            }
        )

    return metrics


def checkpoint_metric(metrics):
    anno_values = [
        metrics[key]
        for key in ("anno_auc1", "anno_auc2")
        if metrics[key] is not None and np.isfinite(metrics[key])
    ]
    if anno_values:
        return max(anno_values)
    return max(metrics["auc1"], metrics["auc2"])


def log_validation(wandb_run, phase, epoch, metrics, metric):
    if wandb_run is None:
        return
    values = {
        f"{phase}/val_{key}": (
            value if value is not None else float("nan")
        )
        for key, value in metrics.items()
    }
    values[f"{phase}/checkpoint_metric"] = metric
    values[f"{phase}/epoch"] = epoch
    wandb_run.log(values)


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    metric,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metric": metric,
        },
        path,
    )


def train_phase1(
    model,
    normal_loader,
    accident_loader,
    test_loader,
    gt,
    prompt_text,
    args,
    device,
    wandb_run,
):
    print("Starting phase 1: weak five-crop MIL")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.phase1_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = MultiStepLR(
        optimizer,
        args.phase1_scheduler_milestones,
        getattr(
            args,
            "phase1_scheduler_rate",
            getattr(args, "scheduler_rate", 0.5),
        ),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "model_best_phase1.pth"
    checkpoint_path = output_dir / "checkpoint_phase1.pth"
    best_metric = float("-inf")
    no_improve = 0

    for epoch in range(args.phase1_epochs):
        model.train()
        totals = {"loss1": 0.0, "loss2": 0.0, "loss3": 0.0, "total": 0.0}
        total_batches = min(len(normal_loader), len(accident_loader))
        if total_batches == 0:
            raise RuntimeError("phase-1 loader produced zero batches")

        normal_iter = iter(normal_loader)
        accident_iter = iter(accident_loader)
        progress = tqdm(
            range(total_batches),
            desc=f"Phase 1 epoch {epoch + 1}/{args.phase1_epochs}",
        )
        for batch_index in progress:
            normal_features, normal_labels, normal_lengths = next(normal_iter)
            accident_features, accident_labels, accident_lengths = next(
                accident_iter
            )
            features = torch.cat([normal_features, accident_features], dim=0)
            lengths = torch.cat([normal_lengths, accident_lengths], dim=0)
            labels = labels_to_indices(
                list(normal_labels) + list(accident_labels),
                device,
            )
            k_values = get_top_k_values(
                lengths.to(device),
                args.top_k_divisor,
                args.top_k_min,
                args.top_k_max,
            )

            text_features, logits1, logits2 = forward_five_crop(
                model,
                features,
                lengths,
                prompt_text,
                args.use_padding_mask,
                device,
            )
            loss1, loss2 = compute_mil_losses(
                logits1,
                logits2,
                labels,
                lengths.to(device),
                k_values,
                args.spatial_top_k,
            )
            loss3 = text_separation_loss(
                text_features,
                args.text_loss_weight,
            )
            total_loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            values = {
                "loss1": loss1,
                "loss2": loss2,
                "loss3": loss3,
                "total": total_loss,
            }
            for name, value in values.items():
                totals[name] += value.item()
            progress.set_postfix(
                {
                    "mil1": f"{totals['loss1'] / (batch_index + 1):.4f}",
                    "mil2": f"{totals['loss2'] / (batch_index + 1):.4f}",
                }
            )

        averages = {
            name: value / total_batches for name, value in totals.items()
        }
        if wandb_run is not None:
            wandb_run.log(
                {
                    **{
                        f"phase1/train_{name}": value
                        for name, value in averages.items()
                    },
                    "phase1/lr": optimizer.param_groups[0]["lr"],
                    "phase1/epoch": epoch + 1,
                }
            )
        print_train_summary(
            "Phase 1",
            epoch + 1,
            args.phase1_epochs,
            averages,
            optimizer.param_groups[0]["lr"],
        )

        metrics = evaluate_five_crop(
            model,
            test_loader,
            prompt_text,
            gt,
            args,
            device,
        )
        metric = checkpoint_metric(metrics)
        print_validation_summary(metrics, metric)
        log_validation(wandb_run, "phase1", epoch + 1, metrics, metric)
        scheduler.step()

        torch.save(
            model.state_dict(),
            output_dir / "model_cur_phase1.pth",
        )
        if args.save_every_epoch:
            torch.save(
                model.state_dict(),
                output_dir / f"phase1_epoch_{epoch + 1}.pth",
            )

        if metric > best_metric + args.early_stop_min_delta:
            best_metric = metric
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                metric,
            )
        else:
            no_improve += 1
            if (
                args.phase1_early_stop_patience > 0
                and no_improve >= args.phase1_early_stop_patience
            ):
                print("Phase 1 early stopping")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"Phase 1 complete. Loaded best model: {best_path}")


def train_phase2(
    model,
    normal_loader,
    timestamp_accident_loader,
    test_loader,
    gt,
    prompt_text,
    args,
    device,
    wandb_run,
):
    print("Starting phase 2: timestamp-supervised five-crop training")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.phase2_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = MultiStepLR(
        optimizer,
        args.phase2_scheduler_milestones,
        getattr(
            args,
            "phase2_scheduler_rate",
            getattr(args, "scheduler_rate", 0.5),
        ),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "model_best_phase2.pth"
    checkpoint_path = output_dir / "checkpoint_phase2.pth"
    best_metric = float("-inf")
    no_improve = 0

    for epoch in range(args.phase2_epochs):
        model.train()
        totals = {
            "loss1": 0.0,
            "loss2": 0.0,
            "loss3": 0.0,
            "temp_a1": 0.0,
            "temp_a2": 0.0,
            "gap_a1": 0.0,
            "gap_a2": 0.0,
            "total": 0.0,
        }
        total_batches = min(
            len(normal_loader),
            len(timestamp_accident_loader),
        )
        if total_batches == 0:
            raise RuntimeError("phase-2 loader produced zero batches")

        normal_iter = iter(normal_loader)
        accident_iter = iter(timestamp_accident_loader)
        progress = tqdm(
            range(total_batches),
            desc=f"Phase 2 epoch {epoch + 1}/{args.phase2_epochs}",
        )
        for batch_index in progress:
            normal_features, normal_labels, normal_lengths = next(normal_iter)
            (
                accident_features,
                accident_labels,
                accident_lengths,
                timestamps,
                seconds_per_snippet,
            ) = next(accident_iter)

            normal_batch_size = normal_features.shape[0]
            features = torch.cat([normal_features, accident_features], dim=0)
            lengths = torch.cat([normal_lengths, accident_lengths], dim=0)
            labels = labels_to_indices(
                list(normal_labels) + list(accident_labels),
                device,
            )
            device_lengths = lengths.to(device)
            k_values = get_top_k_values(
                device_lengths,
                args.top_k_divisor,
                args.top_k_min,
                args.top_k_max,
            )

            text_features, logits1, logits2 = forward_five_crop(
                model,
                features,
                lengths,
                prompt_text,
                args.use_padding_mask,
                device,
            )
            loss1, loss2 = compute_mil_losses(
                logits1,
                logits2,
                labels,
                device_lengths,
                k_values,
                args.spatial_top_k,
            )
            loss3 = text_separation_loss(
                text_features,
                args.text_loss_weight,
            )
            scores_a1, scores_a2 = aggregate_spatial_scores(
                logits1,
                logits2,
                args.spatial_top_k,
            )
            temp_a1, temp_a2, gap_a1, gap_a2 = temporal_supervision(
                scores_a1,
                scores_a2,
                normal_batch_size,
                accident_lengths,
                timestamps,
                seconds_per_snippet,
                k_values[normal_batch_size:],
                args,
            )
            total_loss = (
                args.phase2_mil_weight * (loss1 + loss2)
                + loss3
                + args.temporal_weight * (temp_a1 + temp_a2)
                + args.gap_weight * (gap_a1 + gap_a2)
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            values = {
                "loss1": loss1,
                "loss2": loss2,
                "loss3": loss3,
                "temp_a1": temp_a1,
                "temp_a2": temp_a2,
                "gap_a1": gap_a1,
                "gap_a2": gap_a2,
                "total": total_loss,
            }
            for name, value in values.items():
                totals[name] += value.item()
            progress.set_postfix(
                {
                    "mil1": f"{totals['loss1'] / (batch_index + 1):.4f}",
                    "mil2": f"{totals['loss2'] / (batch_index + 1):.4f}",
                    "temp1": f"{totals['temp_a1'] / (batch_index + 1):.4f}",
                    "temp2": f"{totals['temp_a2'] / (batch_index + 1):.4f}",
                }
            )

        averages = {
            name: value / total_batches for name, value in totals.items()
        }
        if wandb_run is not None:
            wandb_run.log(
                {
                    **{
                        f"phase2/train_{name}": value
                        for name, value in averages.items()
                    },
                    "phase2/lr": optimizer.param_groups[0]["lr"],
                    "phase2/epoch": epoch + 1,
                }
            )
        print_train_summary(
            "Phase 2",
            epoch + 1,
            args.phase2_epochs,
            averages,
            optimizer.param_groups[0]["lr"],
        )

        metrics = evaluate_five_crop(
            model,
            test_loader,
            prompt_text,
            gt,
            args,
            device,
        )
        metric = checkpoint_metric(metrics)
        print_validation_summary(metrics, metric)
        log_validation(wandb_run, "phase2", epoch + 1, metrics, metric)
        scheduler.step()

        torch.save(
            model.state_dict(),
            output_dir / "model_cur_phase2.pth",
        )
        if args.save_every_epoch:
            torch.save(
                model.state_dict(),
                output_dir / f"phase2_epoch_{epoch + 1}.pth",
            )

        if metric > best_metric + args.early_stop_min_delta:
            best_metric = metric
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                metric,
            )
        else:
            no_improve += 1
            if (
                args.phase2_early_stop_patience > 0
                and no_improve >= args.phase2_early_stop_patience
            ):
                print("Phase 2 early stopping")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    torch.save(model.state_dict(), output_dir / "model_final.pth")
    print(f"Phase 2 complete. Final model: {output_dir / 'model_final.pth'}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def validate_args(args):
    if args.classes_num != 2:
        raise ValueError("two-phase training requires exactly 2 classes")
    if args.crop_count != 5:
        raise ValueError("the new feature pipeline requires exactly 5 crops")
    if not 1 <= args.spatial_top_k <= args.crop_count:
        raise ValueError("spatial top-k must be between 1 and crop count")
    if args.phase1_epochs <= 0 or args.phase2_epochs <= 0:
        raise ValueError("phase 1 and phase 2 epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.reference_fps <= 0 or args.frames_per_snippet <= 0:
        raise ValueError("reference FPS and frames per snippet must be positive")
    if args.phase1_lr <= 0 or args.phase2_lr <= 0:
        raise ValueError("learning rates must be positive")
    if not 0 <= args.normal_target < args.accident_target <= 1:
        raise ValueError("targets must satisfy 0 <= normal < accident <= 1")
    if args.pre_normal_buffer_seconds < 0:
        raise ValueError("pre-normal buffer must be non-negative")
    if args.accident_window_end_seconds <= args.accident_window_start_seconds:
        raise ValueError("accident window end must be after its start")
    if args.post_normal_start_seconds <= args.accident_window_end_seconds:
        raise ValueError("post-normal region must start after accident window")
    if not 0 <= args.post_normal_weight <= 1:
        raise ValueError("post-normal weight must be between 0 and 1")
    if min(
        args.phase2_mil_weight,
        args.temporal_weight,
        args.gap_weight,
        args.text_loss_weight,
    ) < 0:
        raise ValueError("loss weights must be non-negative")
    if not 0 <= args.gap_margin <= 1:
        raise ValueError("probability gap margin must be between 0 and 1")

    for path in (
        args.pretrained_path,
        args.timestamp_excel,
        args.train_list,
        args.test_list,
        args.gt_path,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
