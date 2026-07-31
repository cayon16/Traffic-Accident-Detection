"""Frame-level evaluation for the five-crop two-stage VADCLIP model."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import CLIPVAD
from ucf_staged_training_common import (
    FiveCropVideoDataset,
    evaluate_five_crop,
)
from utils.tools import build_label_map, get_prompt_text
import ucf_option_staged_training


def load_model_weights(model, model_path, device):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)


def test(model, test_loader, prompt_text, gt, args, device):
    metrics = evaluate_five_crop(
        model,
        test_loader,
        prompt_text,
        gt,
        args,
        device,
    )

    print("AUC1: ", metrics["auc1"], " AP1: ", metrics["ap1"])
    print("AUC2: ", metrics["auc2"], " AP2:", metrics["ap2"])
    print("AUCF: ", metrics["aucf"], " APF:", metrics["apf"])

    print("FAR1: ", metrics["far1"])
    print("FAR2: ", metrics["far2"])
    print("FARF: ", metrics["farf"])

    if metrics["anno_auc1"] is not None:
        print(
            "Anno AUC1: ",
            metrics["anno_auc1"],
            " Anno AP1: ",
            metrics["anno_ap1"],
        )
        print(
            "Anno AUC2: ",
            metrics["anno_auc2"],
            " Anno AP2: ",
            metrics["anno_ap2"],
        )
        print(
            "Anno AUCF: ",
            metrics["anno_aucf"],
            " Anno APF: ",
            metrics["anno_apf"],
        )
        print(
            "Anno Cohen d1: ",
            metrics["anno_cohen_d1"],
            " Anno Cohen d2: ",
            metrics["anno_cohen_d2"],
        )
    else:
        print("Anno AUC/AP: skipped (only one class in accident frames)")

    return (
        metrics["auc1"],
        metrics["ap1"],
        metrics["auc2"],
        metrics["ap2"],
        metrics["aucf"],
        metrics["apf"],
        metrics["anno_auc1"],
        metrics["anno_ap1"],
        metrics["anno_auc2"],
        metrics["anno_ap2"],
        metrics["far1"],
        metrics["far2"],
        metrics["farf"],
        metrics["anno_cohen_d1"],
        metrics["anno_cohen_d2"],
    )


def validate_args(args):
    if args.classes_num != 2:
        raise ValueError("five-crop testing requires exactly two classes")
    if args.crop_count != 5:
        raise ValueError("five-crop testing requires crop_count=5")
    if not 1 <= args.spatial_top_k <= args.crop_count:
        raise ValueError("spatial_top_k must be in [1, crop_count]")
    for path in (args.test_list, args.gt_path, args.model_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)


def main():
    parser = ucf_option_staged_training.parser
    parser.description = "Five-crop frame-level VADCLIP evaluation"
    parser.add_argument(
        "--model-path",
        required=True,
        help="stage-1 or stage-2 model checkpoint",
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
    gt = np.load(args.gt_path)

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
    model.to(device)

    print("Model:", args.model_path)
    print("Test list:", args.test_list)
    print(
        f"Five-crop aggregation: mean spatial top-{args.spatial_top_k}"
    )
    test(model, test_loader, prompt_text, gt, args, device)


if __name__ == "__main__":
    main()
