from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CLIPVAD
from ucf_staged_training_common import (
    FiveCropVideoDataset,
    aggregate_spatial_scores,
    checkpoint_metric,
    evaluate_five_crop,
    forward_five_crop,
    get_top_k_values,
    labels_to_indices,
    load_timestamp_map,
    log_validation,
    print_train_summary,
    print_validation_summary,
    setup_seed,
    text_separation_loss,
)
from ucf_train_stage1_mil import basic_mil_losses, load_model_weights
from utils.tools import build_label_map, get_prompt_text
import ucf_option_staged_training


def timestamp_gap_losses(
    scores_a1,
    scores_a2,
    accident_lengths,
    timestamps,
    seconds_per_snippet,
    accident_k_values,
    args,
):
    gap_a1 = []
    gap_a2 = []

    for local_index in range(accident_lengths.shape[0]):
        length = int(accident_lengths[local_index])
        timestamp = float(timestamps[local_index])
        snippet_seconds = float(seconds_per_snippet[local_index])
        k_value = int(accident_k_values[local_index])

        pre_end = math.floor(
            (timestamp - args.pre_normal_buffer_seconds) / snippet_seconds
        )
        accident_start = math.floor(
            (timestamp + args.accident_window_start_seconds)
            / snippet_seconds
        )
        accident_end = math.ceil(
            (timestamp + args.accident_window_end_seconds)
            / snippet_seconds
        )
        post_start = math.ceil(
            (timestamp + args.post_normal_start_seconds)
            / snippet_seconds
        )

        pre_end = min(max(pre_end, 0), length)
        accident_start = min(max(accident_start, 0), length - 1)
        accident_end = min(
            max(accident_end, accident_start + 1),
            length,
        )
        post_start = min(max(post_start, accident_end), length)

        device = scores_a1.device
        pre_indices = torch.arange(0, pre_end, device=device)
        accident_indices = torch.arange(
            accident_start,
            accident_end,
            device=device,
        )
        post_indices = torch.arange(post_start, length, device=device)

        for branch_scores, gap_losses in (
            (scores_a1, gap_a1),
            (scores_a2, gap_a2),
        ):
            scores = branch_scores[local_index, :length]
            gap_k = min(
                k_value,
                int(pre_indices.numel() + post_indices.numel()),
                int(accident_indices.numel()),
            )
            if gap_k > 0:
                pre_k = min(
                    (gap_k + 1) // 2,
                    int(pre_indices.numel()),
                )
                post_k = min(
                    gap_k // 2,
                    int(post_indices.numel()),
                )
                remaining = gap_k - pre_k - post_k
                if remaining > 0:
                    extra_pre = min(
                        remaining,
                        int(pre_indices.numel()) - pre_k,
                    )
                    pre_k += extra_pre
                    remaining -= extra_pre
                if remaining > 0:
                    post_k += min(
                        remaining,
                        int(post_indices.numel()) - post_k,
                    )

                normal_parts = []
                if pre_k > 0:
                    normal_parts.append(
                        torch.topk(
                            scores.index_select(0, pre_indices),
                            k=pre_k,
                        ).values
                    )
                if post_k > 0:
                    normal_parts.append(
                        torch.topk(
                            scores.index_select(0, post_indices),
                            k=post_k,
                        ).values
                    )
                hard_normal = torch.cat(normal_parts).sort(
                    descending=True
                ).values
                strong_accident = torch.topk(
                    scores.index_select(0, accident_indices),
                    k=gap_k,
                ).values
                gap_losses.append(
                    F.relu(
                        args.gap_margin
                        + hard_normal
                        - strong_accident
                    ).mean()
                )
            else:
                gap_losses.append(scores.sum() * 0.0)

    zero_a1 = scores_a1.sum() * 0.0
    zero_a2 = scores_a2.sum() * 0.0
    return (
        torch.stack(gap_a1).mean() if gap_a1 else zero_a1,
        torch.stack(gap_a2).mean() if gap_a2 else zero_a2,
    )


def save_checkpoint(path, epoch, model, optimizer, scheduler, metric):
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


def validate_args(args):
    if args.classes_num != 2:
        raise ValueError("stage 2 requires exactly two classes")
    if args.crop_count != 5:
        raise ValueError("stage 2 requires five crops per video")
    if not 1 <= args.spatial_top_k <= args.crop_count:
        raise ValueError("spatial top-k must be in [1, crop_count]")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid DataLoader configuration")
    if args.phase2_epochs <= 0 or args.phase2_lr <= 0:
        raise ValueError("phase-2 epochs and learning rate must be positive")
    if not 0 < args.phase2_scheduler_rate <= 1:
        raise ValueError("phase-2 scheduler rate must be in (0, 1]")
    if args.reference_fps <= 0 or args.frames_per_snippet <= 0:
        raise ValueError("FPS and frames per snippet must be positive")
    if args.top_k_divisor <= 0:
        raise ValueError("top-k divisor must be positive")
    if not 1 <= args.top_k_min <= args.top_k_max:
        raise ValueError("top-k bounds must satisfy 1 <= min <= max")
    if args.pre_normal_buffer_seconds < 0:
        raise ValueError("pre-normal buffer must be non-negative")
    if (
        args.accident_window_end_seconds
        <= args.accident_window_start_seconds
    ):
        raise ValueError("accident window end must be after its start")
    if (
        args.post_normal_start_seconds
        <= args.accident_window_end_seconds
    ):
        raise ValueError("post-normal region must start after accident window")
    if min(
        args.phase2_mil_weight,
        args.gap_weight,
        args.text_loss_weight,
    ) < 0:
        raise ValueError("loss weights must be non-negative")
    if not 0 <= args.gap_margin <= 1:
        raise ValueError("gap margin must be in [0, 1]")
    if args.gap_warmup_epochs < 0:
        raise ValueError("gap warm-up epochs must be non-negative")
    if args.max_grad_norm < 0:
        raise ValueError("max gradient norm must be non-negative")

    required_paths = (
        args.train_list,
        args.test_list,
        args.gt_path,
        args.timestamp_excel,
        args.phase2_pretrained_path,
    )
    for path in required_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if (
        args.phase2_resume_checkpoint
        and not Path(args.phase2_resume_checkpoint).is_file()
    ):
        raise FileNotFoundError(args.phase2_resume_checkpoint)


def train(
    model,
    normal_loader,
    accident_loader,
    test_loader,
    args,
    device,
):
    output_dir = Path(args.phase2_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "model_best.pth"
    current_path = output_dir / "model_current.pth"
    final_path = output_dir / "model_final.pth"
    checkpoint_path = output_dir / "checkpoint.pth"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.phase2_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=args.phase2_scheduler_milestones,
        gamma=args.phase2_scheduler_rate,
    )
    start_epoch = 0
    best_metric = float("-inf")
    no_improve = 0

    if args.phase2_resume_checkpoint:
        checkpoint = torch.load(
            args.phase2_resume_checkpoint,
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["metric"])
        torch.save(model.state_dict(), best_path)
        print("Resumed phase 2 from epoch:", start_epoch)

    label_map = build_label_map(
        args.normal_prompt,
        args.accident_prompt,
    )
    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.phase2_wandb_run_name or None,
            config=vars(args),
        )

    try:
        if not args.phase2_resume_checkpoint:
            baseline_metrics = evaluate_five_crop(
                model,
                test_loader,
                prompt_text,
                gt,
                args,
                device,
            )
            best_metric = checkpoint_metric(baseline_metrics)
            torch.save(model.state_dict(), best_path)
            print("\nPhase-1 baseline before Stage 2")
            print_validation_summary(baseline_metrics, best_metric)
            log_validation(
                wandb_run,
                "stage2",
                0,
                baseline_metrics,
                best_metric,
            )

        for epoch in range(start_epoch, args.phase2_epochs):
            model.train()
            totals = {
                "loss1": 0.0,
                "loss2": 0.0,
                "loss3": 0.0,
                "phase1_loss": 0.0,
                "gap_a1": 0.0,
                "gap_a2": 0.0,
                "gap_weight": 0.0,
                "total": 0.0,
            }
            if args.gap_warmup_epochs > 0:
                gap_warmup = min(
                    1.0,
                    (epoch + 1) / args.gap_warmup_epochs,
                )
            else:
                gap_warmup = 1.0
            current_gap_weight = args.gap_weight * gap_warmup
            total_batches = max(
                len(normal_loader),
                len(accident_loader),
            )
            if total_batches == 0:
                raise RuntimeError("phase-2 loader produced zero batches")

            normal_iterator = iter(normal_loader)
            accident_iterator = iter(accident_loader)
            progress = tqdm(
                range(total_batches),
                desc=f"Stage 2 epoch {epoch + 1}/{args.phase2_epochs}",
            )
            for batch_index in progress:
                try:
                    normal_batch = next(normal_iterator)
                except StopIteration:
                    normal_iterator = iter(normal_loader)
                    normal_batch = next(normal_iterator)
                (
                    normal_features,
                    normal_labels,
                    normal_lengths,
                ) = normal_batch

                try:
                    accident_batch = next(accident_iterator)
                except StopIteration:
                    accident_iterator = iter(accident_loader)
                    accident_batch = next(accident_iterator)
                (
                    accident_features,
                    accident_labels,
                    accident_lengths,
                    timestamps,
                    seconds_per_snippet,
                ) = accident_batch

                normal_batch_size = normal_features.shape[0]
                features = torch.cat(
                    [normal_features, accident_features],
                    dim=0,
                )
                lengths = torch.cat(
                    [normal_lengths, accident_lengths],
                    dim=0,
                ).to(device)
                labels = labels_to_indices(
                    list(normal_labels) + list(accident_labels),
                    device,
                )
                k_values = get_top_k_values(
                    lengths,
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
                loss1, loss2 = basic_mil_losses(
                    logits1,
                    logits2,
                    labels,
                    lengths,
                    k_values,
                    args.spatial_top_k,
                )
                loss3 = text_separation_loss(
                    text_features,
                    args.text_loss_weight,
                )
                phase1_loss = loss1 + loss2 + loss3

                scores_a1, scores_a2 = aggregate_spatial_scores(
                    logits1,
                    logits2,
                    args.spatial_top_k,
                )
                timestamp_mask = torch.isfinite(timestamps)
                if timestamp_mask.any():
                    timestamp_mask_device = timestamp_mask.to(device)
                    gap_a1, gap_a2 = timestamp_gap_losses(
                        scores_a1[normal_batch_size:][
                            timestamp_mask_device
                        ],
                        scores_a2[normal_batch_size:][
                            timestamp_mask_device
                        ],
                        accident_lengths.to(device)[
                            timestamp_mask_device
                        ],
                        timestamps[timestamp_mask],
                        seconds_per_snippet[timestamp_mask],
                        k_values[normal_batch_size:][
                            timestamp_mask_device
                        ],
                        args,
                    )
                else:
                    gap_a1 = scores_a1.sum() * 0.0
                    gap_a2 = scores_a2.sum() * 0.0

                gap_loss = current_gap_weight * (gap_a1 + gap_a2)
                total_loss = (
                    args.phase2_mil_weight * phase1_loss
                    + gap_loss
                )

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )
                optimizer.step()

                values = {
                    "loss1": loss1,
                    "loss2": loss2,
                    "loss3": loss3,
                    "phase1_loss": phase1_loss,
                    "gap_a1": gap_a1,
                    "gap_a2": gap_a2,
                    "gap_weight": current_gap_weight,
                    "total": total_loss,
                }
                for name, value in values.items():
                    totals[name] += (
                        value.item()
                        if torch.is_tensor(value)
                        else value
                    )
                progress.set_postfix(
                    phase1=(
                        f"{totals['phase1_loss'] / (batch_index + 1):.4f}"
                    ),
                    gap1=f"{totals['gap_a1'] / (batch_index + 1):.4f}",
                    gap2=f"{totals['gap_a2'] / (batch_index + 1):.4f}",
                )

            averages = {
                name: value / total_batches
                for name, value in totals.items()
            }
            print_train_summary(
                "Stage 2",
                epoch + 1,
                args.phase2_epochs,
                averages,
                optimizer.param_groups[0]["lr"],
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{
                            f"stage2/train_{name}": value
                            for name, value in averages.items()
                        },
                        "stage2/lr": optimizer.param_groups[0]["lr"],
                        "stage2/epoch": epoch + 1,
                    }
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
            log_validation(
                wandb_run,
                "stage2",
                epoch + 1,
                metrics,
                metric,
            )
            scheduler.step()
            torch.save(model.state_dict(), current_path)

            if args.save_every_epoch:
                torch.save(
                    model.state_dict(),
                    output_dir / f"epoch_{epoch + 1}.pth",
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
                    print("Stage 2 early stopping")
                    break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    if not best_path.is_file():
        raise RuntimeError("stage 2 did not produce a valid best model")
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), final_path)
    print("Stage 2 complete. Final model:", final_path)


def main():
    args = ucf_option_staged_training.parser.parse_args()
    validate_args(args)
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp_map = load_timestamp_map(args.timestamp_excel)
    base_snippet_seconds = (
        args.frames_per_snippet / args.reference_fps
    )
    normal_dataset = FiveCropVideoDataset(
        args.visual_length,
        args.train_list,
        args.crop_count,
        label_filter="normal",
    )
    accident_dataset = FiveCropVideoDataset(
        args.visual_length,
        args.train_list,
        args.crop_count,
        label_filter="accident",
        timestamp_map=timestamp_map,
        base_snippet_seconds=base_snippet_seconds,
        timestamp_optional=True,
    )
    test_dataset = FiveCropVideoDataset(
        args.visual_length,
        args.test_list,
        args.crop_count,
        test_mode=True,
    )
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    accident_loader = DataLoader(
        accident_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
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
    ).to(device)
    load_model_weights(
        model,
        args.phase2_pretrained_path,
        device,
        strict=True,
    )
    train(
        model,
        normal_loader,
        accident_loader,
        test_loader,
        args,
        device,
    )


if __name__ == "__main__":
    main()
