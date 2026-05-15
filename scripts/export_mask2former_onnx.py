import argparse
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForUniversalSegmentation


class Mask2FormerOnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs.class_queries_logits, outputs.masks_queries_logits


def export_model(model_dir, output, image_size, opset):
    model = AutoModelForUniversalSegmentation.from_pretrained(model_dir).eval()
    wrapper = Mask2FormerOnnxWrapper(model).eval()

    pixel_values = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    pixel_mask = torch.ones(1, image_size, image_size, dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (pixel_values, pixel_mask),
        str(output),
        input_names=["pixel_values", "pixel_mask"],
        output_names=["class_queries_logits", "masks_queries_logits"],
        opset_version=opset,
        dynamo=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Export Mask2Former line detector to ONNX.")
    parser.add_argument("--model-dir", default="mask2former_lines_scratch/best")
    parser.add_argument("--output", default="mask2former_lines_scratch/best/model_512_dynamo.onnx")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=20)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_model(args.model_dir, output, args.image_size, args.opset)
    print(output)


if __name__ == "__main__":
    main()
