from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CLIPVAD
from ucf_staged_training_common import (
    FiveCropVideoDataset,
    checkpoint_metric,
    evaluate_five_crop,
    forward_five_crop,
    get_top_k_values,
    labels_to_indices,
    log_validation,
    print_train_summary,
    print_validation_summary,
    setup_seed,
    text_separation_loss,
)
from utils.tools import build_label_map, get_prompt_text
import ucf_option_staged_training


def temporal_top_k_mean(values, ranking_scores, lengths, k_values):
    """Pool a variable temporal top-k without looping over the batch."""
    temporal_length = ranking_scores.shape[1]
    valid_mask = (
        torch.arange(temporal_length, device=ranking_scores.device)
        .unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    masked_ranking = ranking_scores.masked_fill(
        ~valid_mask,
        torch.finfo(ranking_scores.dtype).min,
    )
    max_k = int(k_values.max().item())
    top_indices = torch.topk(
        masked_ranking,
        k=max_k,
        dim=1,
    ).indices
    selected_mask = (
        torch.arange(max_k, device=k_values.device).unsqueeze(0)
        < k_values.unsqueeze(1)
    )

    if values.ndim == 2:
        selected_values = torch.gather(values, 1, top_indices)
        return (
            selected_values * selected_mask.to(values.dtype)
        ).sum(dim=1) / k_values.to(values.dtype)

    gather_indices = top_indices.unsqueeze(-1).expand(
        -1,
        -1,
        values.shape[-1],
    )
    selected_values = torch.gather(values, 1, gather_indices)
    return (
        selected_values
        * selected_mask.unsqueeze(-1).to(values.dtype)
    ).sum(dim=1) / k_values.unsqueeze(-1).to(values.dtype)


def basic_mil_losses(
    logits1,
    logits2,
    labels,
    lengths,
    k_values,
    spatial_top_k,
):
    """Apply spatial crop pooling before temporal MIL for each video."""
    scores_a1 = torch.sigmoid(logits1.squeeze(-1))
    spatial_scores_a1 = torch.topk(
        scores_a1,
        k=spatial_top_k,
        dim=1,
    ).values.mean(dim=1)
    video_scores_a1 = temporal_top_k_mean(
        spatial_scores_a1,
        spatial_scores_a1,
        lengths,
        k_values,
    )

    crop_margins_a2 = logits2[..., 1] - logits2[..., 0]
    spatial_crop_indices = torch.topk(
        crop_margins_a2,
        k=spatial_top_k,
        dim=1,
    ).indices
    gather_indices = spatial_crop_indices.unsqueeze(-1).expand(
        -1,
        -1,
        -1,
        logits2.shape[-1],
    )
    spatial_logits_a2 = torch.gather(
        logits2,
        1,
        gather_indices,
    ).mean(dim=1)
    spatial_margins_a2 = (
        spatial_logits_a2[..., 1] - spatial_logits_a2[..., 0]
    )
    video_logits_a2 = temporal_top_k_mean(
        spatial_logits_a2,
        spatial_margins_a2,
        lengths,
        k_values,
    )

    loss1 = F.binary_cross_entropy(
        video_scores_a1,
        labels.to(logits1.dtype),
    )
    loss2 = F.cross_entropy(video_logits_a2, labels)
    return loss1, loss2


def load_model_weights(model, path, device, strict):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print("Loaded model:", path)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)


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
        raise ValueError("stage 1 requires exactly two classes")
    if args.crop_count != 5:
        raise ValueError("stage 1 requires five crops per video")
    if not 1 <= args.spatial_top_k <= args.crop_count:
        raise ValueError("spatial top-k must be in [1, crop_count]")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid DataLoader configuration")
    if args.phase1_epochs <= 0 or args.phase1_lr <= 0:
        raise ValueError("phase-1 epochs and learning rate must be positive")
    if not 0 < args.phase1_scheduler_rate <= 1:
        raise ValueError("phase-1 scheduler rate must be in (0, 1]")
    if args.top_k_divisor <= 0:
        raise ValueError("top-k divisor must be positive")
    if not 1 <= args.top_k_min <= args.top_k_max:
        raise ValueError("top-k bounds must satisfy 1 <= min <= max")
    if args.text_loss_weight < 0:
        raise ValueError("text loss weight must be non-negative")

    required_paths = (
        args.train_list,
        args.test_list,
        args.gt_path,
        args.phase1_pretrained_path,
    )
    for path in required_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if (
        args.phase1_resume_checkpoint
        and not Path(args.phase1_resume_checkpoint).is_file()
    ):
        raise FileNotFoundError(args.phase1_resume_checkpoint)


def train(model, normal_loader, accident_loader, test_loader, args, device):
    output_dir = Path(args.phase1_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "model_best.pth"
    current_path = output_dir / "model_current.pth"
    checkpoint_path = output_dir / "checkpoint.pth"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.phase1_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=args.phase1_scheduler_milestones,
        gamma=args.phase1_scheduler_rate,
    )
    start_epoch = 0
    best_metric = float("-inf")
    no_improve = 0

    if args.phase1_resume_checkpoint:
        checkpoint = torch.load(
            args.phase1_resume_checkpoint,
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["metric"])
        torch.save(model.state_dict(), best_path)
        print("Resumed phase 1 from epoch:", start_epoch)

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
            name=args.phase1_wandb_run_name or None,
            config=vars(args),
        )

    try:
        for epoch in range(start_epoch, args.phase1_epochs):
            model.train()
            totals = {
                "loss1": 0.0,
                "loss2": 0.0,
                "loss3": 0.0,
                "total": 0.0,
            }
            total_batches = min(len(normal_loader), len(accident_loader))
            if total_batches == 0:
                raise RuntimeError("phase-1 loader produced zero batches")

            normal_iterator = iter(normal_loader)
            accident_iterator = iter(accident_loader)
            progress = tqdm(
                range(total_batches),
                desc=f"Stage 1 epoch {epoch + 1}/{args.phase1_epochs}",
            )
            for batch_index in progress:
                normal_features, normal_labels, normal_lengths = next(
                    normal_iterator
                )
                accident_features, accident_labels, accident_lengths = next(
                    accident_iterator
                )
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
                    mil1=f"{totals['loss1'] / (batch_index + 1):.4f}",
                    mil2=f"{totals['loss2'] / (batch_index + 1):.4f}",
                )

            averages = {
                name: value / total_batches
                for name, value in totals.items()
            }
            print_train_summary(
                "Stage 1",
                epoch + 1,
                args.phase1_epochs,
                averages,
                optimizer.param_groups[0]["lr"],
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{
                            f"stage1/train_{name}": value
                            for name, value in averages.items()
                        },
                        "stage1/lr": optimizer.param_groups[0]["lr"],
                        "stage1/epoch": epoch + 1,
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
                "stage1",
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
                    args.phase1_early_stop_patience > 0
                    and no_improve >= args.phase1_early_stop_patience
                ):
                    print("Stage 1 early stopping")
                    break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    if not best_path.is_file():
        raise RuntimeError("stage 1 did not produce a valid best model")
    print("Stage 1 complete. Best model:", best_path)


def main():
    args = ucf_option_staged_training.parser.parse_args()
    validate_args(args)
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        drop_last=True,
        num_workers=args.num_workers,
    )
    accident_loader = DataLoader(
        accident_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
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
        args.phase1_pretrained_path,
        device,
        strict=False,
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
