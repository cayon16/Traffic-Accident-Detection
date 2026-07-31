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

from clip import clip


DEFAULT_OUTPUT_PATH = str(PROJECT_DIR / "models" / "clip_vit_b16_image.onnx")


class CLIPImageEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(images).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CLIP ViT-B/16 image encoder to ONNX."
    )
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--clip-name", default="ViT-B/16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dynamic-batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def verify_onnx(
    onnx_path: Path,
    wrapper: CLIPImageEncoder,
    images: torch.Tensor,
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
    ort_features = session.run(
        None,
        {"images": images.detach().cpu().numpy().astype(np.float32)},
    )[0]

    with torch.no_grad():
        pt_features = wrapper(images).detach().cpu().numpy()

    diff = np.max(np.abs(ort_features - pt_features))
    print(f"ONNX check passed. max_abs_diff features={diff:.6f}")
    print(f"ONNX Runtime providers: {session.get_providers()}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading CLIP image encoder: {args.clip_name}")
    clip_model, _ = clip.load(args.clip_name, device=device)
    clip_model.float()
    clip_model.eval()

    wrapper = CLIPImageEncoder(clip_model).to(device)
    wrapper.eval()

    images = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        dtype=torch.float32,
        device=device,
    )

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"},
            "features": {0: "batch"},
        }

    print(f"Exporting ONNX: {output_path}")
    torch.onnx.export(
        wrapper,
        (images,),
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["features"],
        dynamic_axes=dynamic_axes,
    )

    print(f"Done. ONNX model saved to: {output_path}")
    if args.dynamic_batch:
        print("ONNX batch size: dynamic")
    else:
        print(f"Fixed ONNX batch size: {args.batch_size}")

    if not args.skip_verify:
        verify_onnx(output_path, wrapper, images)


if __name__ == "__main__":
    main()
