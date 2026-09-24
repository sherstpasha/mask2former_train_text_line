"""Export a trained Mask2Former checkpoint to ONNX."""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForUniversalSegmentation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "mask2former_lines_swin_small_page_medium_ft100" / "best_f1"


class Mask2FormerOnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs.class_queries_logits, outputs.masks_queries_logits


def export_model(model_dir, output_path, image_size, opset, fp16=False):
    dtype = torch.float16 if fp16 else torch.float32
    model = AutoModelForUniversalSegmentation.from_pretrained(model_dir).cpu().eval().to(dtype=dtype)
    wrapper = Mask2FormerOnnxWrapper(model).eval()
    pixel_values = torch.randn(1, 3, image_size, image_size, dtype=dtype)
    pixel_mask = torch.ones(1, image_size, image_size, dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (pixel_values, pixel_mask),
        str(output_path),
        input_names=["pixel_values", "pixel_mask"],
        output_names=["class_queries_logits", "masks_queries_logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "pixel_mask": {0: "batch"},
            "class_queries_logits": {0: "batch"},
            "masks_queries_logits": {0: "batch"},
        },
        opset_version=opset,
        dynamo=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=20)
    parser.add_argument("--fp16", action="store_true", help="Export weights and pixel input as float16.")
    return parser.parse_args()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    output_path = args.output or model_dir / ("model.fp16.onnx" if args.fp16 else "model.onnx")
    output_path = output_path.expanduser().resolve()
    export_model(model_dir, output_path, args.image_size, args.opset, fp16=args.fp16)
    print(f"ONNX model saved to: {output_path}")


if __name__ == "__main__":
    main()
