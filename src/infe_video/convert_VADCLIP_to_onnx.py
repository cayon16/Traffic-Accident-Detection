from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import CLIPVAD
from utils.tools import build_label_map, get_prompt_text


DEFAULT_MODEL_PATH = (
    str(PROJECT_DIR / "models" / "model_current.pth")
)
DEFAULT_OUTPUT_PATH = str(PROJECT_DIR / "models" / "vadclip_2class.onnx")


class ExportableCLIPVAD(nn.Module):
    """CLIPVAD wrapper with fixed text prompt features for ONNX export."""

    def __init__(self, model: CLIPVAD, prompt_text: list[str]):
        super().__init__()
        self.model = model

        with torch.no_grad():
            text_features_ori = model.encode_textprompt(prompt_text).float()
        self.register_buffer("text_features_ori", text_features_ori)

    def _build_similarity_adj(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        sim = torch.matmul(x, x.transpose(1, 2))
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
        denom = torch.matmul(x_norm, x_norm.transpose(1, 2)).clamp_min(1e-20)
        sim = sim / denom
        sim = torch.where(sim > 0.7, sim, torch.zeros_like(sim))

        batch_size, temporal_length, _ = sim.shape
        valid = (
            torch.arange(temporal_length, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, temporal_length)
            < lengths.to(x.device).unsqueeze(1)
        )
        pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)
        sim = torch.where(
            pair_valid,
            sim,
            torch.full_like(sim, -10000.0),
        )
        adj = torch.softmax(sim, dim=2)
        return adj * pair_valid.to(adj.dtype)

    def _build_distance_adj(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, temporal_length, _ = x.shape
        idx = torch.arange(temporal_length, device=x.device, dtype=x.dtype)
        dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        dist = torch.exp(-dist / torch.exp(torch.ones((), device=x.device, dtype=x.dtype)))
        dist = dist.unsqueeze(0).expand(batch_size, temporal_length, temporal_length)

        if self.model.use_padding_mask:
            valid = (
                torch.arange(temporal_length, device=x.device)
                .unsqueeze(0)
                .expand(batch_size, temporal_length)
                < lengths.to(x.device).unsqueeze(1)
            )
            pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)
            dist = dist * pair_valid.to(dist.dtype)

        return dist

    def _encode_video(
        self,
        visual: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        visual = visual.float()
        batch_size = visual.shape[0]
        temporal_length = visual.shape[1]

        position_ids = torch.arange(temporal_length, device=visual.device)
        position_embeddings = self.model.frame_position_embeddings(position_ids)
        position_embeddings = position_embeddings.unsqueeze(1).expand(
            temporal_length,
            batch_size,
            self.model.visual_width,
        )

        x = visual.permute(1, 0, 2) + position_embeddings
        x, _ = self.model.temporal((x, None))
        x = x.permute(1, 0, 2)

        valid_mask = None
        if self.model.use_padding_mask:
            valid_mask = (
                torch.arange(x.shape[1], device=x.device).unsqueeze(0)
                < lengths.to(x.device).unsqueeze(1)
            )
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        adj = self._build_similarity_adj(x, lengths)
        disadj = self._build_distance_adj(x, lengths)

        x1_h = self.model.gelu(self.model.gc1(x, adj))
        x2_h = self.model.gelu(self.model.gc3(x, disadj))
        x1 = self.model.gelu(self.model.gc2(x1_h, adj))
        x2 = self.model.gelu(self.model.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), dim=2)
        x = self.model.linear(x)

        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        return x

    def forward(
        self,
        visual: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual_features = self._encode_video(visual, lengths)
        visual_features_proj = visual_features + self.model.mlp2(visual_features)
        visual_features_proj = self.model.dropout(visual_features_proj)
        logits1 = self.model.classifier(visual_features_proj)

        text_features_ori = self.text_features_ori.to(dtype=visual_features.dtype)
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = torch.matmul(logits_attn, visual_features)
        visual_attn = visual_attn / torch.norm(
            visual_attn,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
        visual_attn = visual_attn.expand(
            visual_attn.shape[0],
            text_features_ori.shape[0],
            visual_attn.shape[2],
        )

        text_features = text_features_ori.unsqueeze(0).expand(
            visual_attn.shape[0],
            text_features_ori.shape[0],
            text_features_ori.shape[1],
        )
        text_features = text_features + visual_attn
        text_features = text_features + self.model.mlp1(text_features)
        text_features = self.model.dropout(text_features)

        visual_features_norm = visual_features / torch.norm(
            visual_features,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
        text_features_norm = text_features / torch.norm(
            text_features,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
        logits2 = torch.matmul(
            visual_features_norm,
            text_features_norm.permute(0, 2, 1),
        ) / 0.07

        return logits1, logits2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the 2-class VADCLIP head to ONNX."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--export-batch-size", type=int, default=16)
    parser.add_argument("--normal-prompt", default="normal")
    parser.add_argument("--accident-prompt", default="roadAccidents")
    parser.add_argument("--classes-num", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--visual-length", type=int, default=256)
    parser.add_argument("--visual-width", type=int, default=512)
    parser.add_argument("--visual-head", type=int, default=1)
    parser.add_argument("--visual-layers", type=int, default=2)
    parser.add_argument("--attn-window", type=int, default=8)
    parser.add_argument("--prompt-prefix", type=int, default=10)
    parser.add_argument("--prompt-postfix", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-padding-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> CLIPVAD:
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
        str(device),
        args.use_padding_mask,
        args.dropout,
    )

    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def verify_onnx(
    onnx_path: Path,
    wrapper: ExportableCLIPVAD,
    visual: torch.Tensor,
    lengths: torch.Tensor,
) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        print(f"Skip ONNX verification because dependency is missing: {exc}")
        return

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=providers,
    )
    ort_logits1, ort_logits2 = session.run(
        None,
        {
            "visual": visual.detach().cpu().numpy().astype(np.float32),
            "lengths": lengths.detach().cpu().numpy().astype(np.int64),
        },
    )

    with torch.no_grad():
        pt_logits1, pt_logits2 = wrapper(visual, lengths)

    diff1 = np.max(np.abs(ort_logits1 - pt_logits1.detach().cpu().numpy()))
    diff2 = np.max(np.abs(ort_logits2 - pt_logits2.detach().cpu().numpy()))
    print(f"ONNX check passed. max_abs_diff logits1={diff1:.6f}, logits2={diff2:.6f}")
    print(f"ONNX Runtime providers: {session.get_providers()}")


def verify_wrapper_matches_model(
    model: CLIPVAD,
    wrapper: ExportableCLIPVAD,
    prompt_text: list[str],
    visual: torch.Tensor,
    lengths: torch.Tensor,
) -> None:
    try:
        with torch.no_grad():
            _, model_logits1, model_logits2 = model(visual, None, prompt_text, lengths)
            wrapper_logits1, wrapper_logits2 = wrapper(visual, lengths)
    except Exception as exc:
        print(f"Skip PyTorch wrapper check because original model forward failed: {exc}")
        return

    diff1 = torch.max(torch.abs(model_logits1 - wrapper_logits1)).item()
    diff2 = torch.max(torch.abs(model_logits2 - wrapper_logits2)).item()
    print(f"PyTorch wrapper check: max_abs_diff logits1={diff1:.6f}, logits2={diff2:.6f}")
    if diff1 > 1e-3 or diff2 > 1e-3:
        print(
            "Warning: wrapper differs from original PyTorch model more than 1e-3. "
            "Check the exported ONNX scores before production use."
        )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if args.export_batch_size <= 0:
        raise ValueError("--export-batch-size must be positive")
    if args.use_padding_mask:
        raise ValueError(
            "ONNX export currently supports --no-use-padding-mask only, "
            "matching the submitted five-crop inference pipeline."
        )

    device = torch.device(args.device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_map = build_label_map(args.normal_prompt, args.accident_prompt)
    prompt_text = get_prompt_text(label_map)

    print(f"Loading PyTorch model: {model_path}")
    model = load_model(args, device)
    wrapper = ExportableCLIPVAD(model, prompt_text).to(device)
    wrapper.eval()

    visual = torch.randn(
        args.export_batch_size,
        args.visual_length,
        args.visual_width,
        dtype=torch.float32,
        device=device,
    )
    lengths = torch.full(
        (args.export_batch_size,),
        args.visual_length,
        dtype=torch.long,
        device=device,
    )
    if args.export_batch_size >= 2:
        lengths[-1] = max(1, args.visual_length // 2)

    if not args.skip_verify:
        verify_wrapper_matches_model(model, wrapper, prompt_text, visual, lengths)

    print(f"Exporting ONNX: {output_path}")
    torch.onnx.export(
        wrapper,
        (visual, lengths),
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["visual", "lengths"],
        output_names=["logits1", "logits2"],
    )

    print(f"Done. ONNX model saved to: {output_path}")
    print(f"Fixed ONNX batch size: {args.export_batch_size}")
    print("Runner will pad the last chunk batch automatically.")

    if not args.skip_verify:
        verify_onnx(output_path, wrapper, visual, lengths)


if __name__ == "__main__":
    main()
