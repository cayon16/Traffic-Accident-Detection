"""Video-level evaluation for the five-crop two-stage VADCLIP model."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from model import CLIPVAD
from ucf_staged_training_common import (
    FiveCropVideoDataset,
    aggregate_spatial_a2_logits,
    aggregate_spatial_scores,
    get_top_k_values,
)
from utils.tools import build_label_map, get_batch_mask, get_prompt_text
import ucf_option_staged_training


def load_model_weights(model, model_path, device):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)


def infer_five_crop_video(
    model,
    features,
    chunk_lengths,
    raw_length,
    prompt_text,
    args,
    device,
):
    crop_count, chunk_count, temporal_length, feature_dim = features.shape
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
        1,
        crop_count,
        chunk_count * temporal_length,
        1,
    )[:, :, :raw_length]
    logits2 = logits2.reshape(
        1,
        crop_count,
        chunk_count * temporal_length,
        2,
    )[:, :, :raw_length]

    scores_a1, scores_a2 = aggregate_spatial_scores(
        logits1,
        logits2,
        args.spatial_top_k,
    )
    spatial_logits_a2 = aggregate_spatial_a2_logits(
        logits2,
        args.spatial_top_k,
    )

    length = torch.tensor([raw_length], device=device)
    k_value = int(
        get_top_k_values(
            length,
            args.top_k_divisor,
            args.top_k_min,
            args.top_k_max,
        )[0]
    )
    video_score_a1 = torch.topk(
        scores_a1[0, :raw_length],
        k=k_value,
    ).values.mean()

    valid_logits_a2 = spatial_logits_a2[0, :raw_length]
    anomaly_margin = valid_logits_a2[:, 1] - valid_logits_a2[:, 0]
    temporal_indices = torch.topk(
        anomaly_margin,
        k=k_value,
    ).indices
    video_logits_a2 = valid_logits_a2.index_select(
        0,
        temporal_indices,
    ).mean(dim=0)
    video_score_a2 = torch.softmax(video_logits_a2, dim=-1)[1]

    return (
        float(video_score_a1),
        float(video_score_a2),
        k_value,
    )


def test_video_level(
    model,
    test_loader,
    prompt_text,
    device,
    save_dir,
    args,
):
    model.to(device)
    model.eval()

    y_true = []
    results = {
        "Branch 1": {"binary": [], "continuous": []},
        "Branch 2": {"binary": [], "continuous": []},
        "Fusion": {"binary": [], "continuous": []},
    }
    temporal_k_values = []

    with torch.inference_mode():
        for features, labels, chunk_lengths, raw_lengths in test_loader:
            features = features.squeeze(0)
            chunk_lengths = chunk_lengths.squeeze(0)
            raw_length = int(raw_lengths[0])
            label_text = str(labels[0])
            gt_label = (
                0 if label_text.strip().casefold() == "normal" else 1
            )
            y_true.append(gt_label)

            score_a1, score_a2, temporal_k = infer_five_crop_video(
                model,
                features,
                chunk_lengths,
                raw_length,
                prompt_text,
                args,
                device,
            )
            temporal_k_values.append(temporal_k)
            score_fused = (score_a1 + score_a2) / 2.0

            for name, score in (
                ("Branch 1", score_a1),
                ("Branch 2", score_a2),
                ("Fusion", score_fused),
            ):
                results[name]["continuous"].append(score)
                results[name]["binary"].append(
                    1 if score >= args.score_thresh else 0
                )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 55)
    print("   FIVE-CROP VIDEO-LEVEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"Total Videos Evaluated : {len(y_true)}")
    print(
        "Spatial rule          : "
        f"Mean of top-{args.spatial_top_k}/{args.crop_count} crops"
    )
    print(
        "Temporal rule         : "
        f"k=clamp(length//{args.top_k_divisor}+1, "
        f"{args.top_k_min}, {args.top_k_max})"
    )
    print(
        "Temporal k range      : "
        f"{min(temporal_k_values)}-{max(temporal_k_values)}"
    )
    print(
        f"Decision rule         : Video score >= "
        f"{args.score_thresh} -> Accident"
    )
    print("=" * 55)

    output_metrics = {}
    for name, data in results.items():
        y_pred = data["binary"]
        y_score = data["continuous"]
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
        else:
            auc = 0.0
            ap = 0.0
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        output_metrics[name] = {
            "auc": auc,
            "ap": ap,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        print(f"\n[{name.upper()}] METRICS:")
        print("-" * 30)
        print(f"ROC AUC   : {auc:.4f}")
        print(f"AP / PR   : {ap:.4f} ({ap * 100:.2f}%)")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        if len(set(y_true)) <= 1:
            continue

        safe_name = name.lower().replace(" ", "")
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            display_labels=["Normal", "Accident"],
            cmap="Blues",
            ax=ax,
        )
        plt.title(
            f"Five-Crop Confusion Matrix "
            f"({name} | Thresh: {args.score_thresh})"
        )
        plt.savefig(
            save_dir / f"confusion_matrix_5crop_{safe_name}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        roc_fig = go.Figure()
        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines+markers",
                name="ROC",
                customdata=roc_thresholds,
                hovertemplate=(
                    "FPR=%{x:.4f}<br>TPR=%{y:.4f}<br>"
                    "threshold=%{customdata:.6f}<extra></extra>"
                ),
            )
        )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"dash": "dash", "color": "gray"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        roc_fig.update_layout(
            title=f"Five-Crop ROC AUC Curve ({name})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        pio.write_html(
            roc_fig,
            file=save_dir / f"roc_curve_5crop_{safe_name}.html",
            auto_open=False,
        )

        precision_values, recall_values, thresholds = (
            precision_recall_curve(y_true, y_score)
        )
        padded_thresholds = np.concatenate([thresholds, [np.nan]])
        pr_fig = go.Figure()
        pr_fig.add_trace(
            go.Scatter(
                x=recall_values,
                y=precision_values,
                mode="lines+markers",
                name="Precision-Recall",
                customdata=padded_thresholds,
                hovertemplate=(
                    "Recall=%{x:.4f}<br>Precision=%{y:.4f}<br>"
                    "threshold=%{customdata:.6f}<extra></extra>"
                ),
            )
        )
        pr_fig.update_layout(
            title=f"Five-Crop Precision-Recall Curve ({name})",
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        pio.write_html(
            pr_fig,
            file=save_dir / f"precision_recall_5crop_{safe_name}.html",
            auto_open=False,
        )

    print("\n" + "=" * 55)
    print(f"All plots have been saved successfully in: {save_dir}")
    print("=" * 55 + "\n")
    return output_metrics


def validate_args(args):
    if args.classes_num != 2:
        raise ValueError("five-crop testing requires exactly two classes")
    if args.crop_count != 5:
        raise ValueError("five-crop testing requires crop_count=5")
    if not 1 <= args.spatial_top_k <= args.crop_count:
        raise ValueError("spatial_top_k must be in [1, crop_count]")
    if args.top_k_divisor <= 0:
        raise ValueError("top_k_divisor must be positive")
    if not 1 <= args.top_k_min <= args.top_k_max:
        raise ValueError("invalid temporal top-k bounds")
    if not 0 <= args.score_thresh <= 1:
        raise ValueError("score_thresh must be in [0, 1]")
    for path in (args.test_list, args.model_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)


def main():
    parser = ucf_option_staged_training.parser
    parser.description = "Five-crop video-level VADCLIP evaluation"
    parser.add_argument(
        "--model-path",
        required=True,
        help="stage-1 or stage-2 model checkpoint",
    )
    parser.add_argument(
        "--score-thresh",
        "--score_thresh",
        dest="score_thresh",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="plot output directory; defaults to the model directory",
    )
    args = parser.parse_args()
    validate_args(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_map = build_label_map(
        args.normal_prompt,
        args.accident_prompt,
    )
    prompt_text = get_prompt_text(label_map)
    test_dataset = FiveCropVideoDataset(
        args.visual_length,
        args.test_list,
        args.crop_count,
        test_mode=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = CLIPVAD(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        device,
        args.use_padding_mask,
        args.dropout,
    )
    load_model_weights(model, args.model_path, device)
    save_dir = args.save_dir or str(Path(args.model_path).parent)
    test_video_level(
        model,
        test_loader,
        prompt_text,
        device,
        save_dir,
        args,
    )


if __name__ == "__main__":
    main()
