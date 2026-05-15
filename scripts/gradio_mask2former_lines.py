import argparse
import sys
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference import crop_line, detect_image, draw_masks, resize_for_inference


DEFAULT_MODEL_DIR = "mask2former_lines_scratch/best"


class Mask2FormerLineDemo:
    def __init__(self, model_dir, cpu=False):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir)
        self.model = AutoModelForUniversalSegmentation.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image, image_size, threshold, mask_threshold, load_max_side, crop_padding):
        if image is None:
            return None, [], "Load an image."

        image = Image.fromarray(image).convert("RGB")
        image = resize_for_inference(image, int(load_max_side))
        detections = detect_image(
            self.model,
            self.processor,
            image,
            self.device,
            int(image_size),
            float(threshold),
            float(mask_threshold),
        )
        overlay = draw_masks(image, detections, numbered=True)
        crops = [
            (crop_line(image, mask, int(crop_padding)), f"line {index}: {score:.2f}")
            for index, (mask, score) in enumerate(detections, start=1)
        ]
        return overlay, crops, f"Found lines: {len(detections)}"


def build_demo(model_dir, cpu=False):
    detector = Mask2FormerLineDemo(model_dir=model_dir, cpu=cpu)

    with gr.Blocks(title="Mask2Former text line masks") as demo:
        gr.Markdown("# Mask2Former text line masks")
        gr.Markdown(f"Model: `{model_dir}`. Device: `{detector.device}`.")

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="numpy", label="Input image")
                run = gr.Button("Find lines", variant="primary")

                with gr.Accordion("Parameters", open=False):
                    image_size = gr.Slider(256, 1024, value=512, step=32, label="Inference size")
                    load_max_side = gr.Slider(512, 4096, value=1600, step=64, label="Input max side")
                    threshold = gr.Slider(0.01, 0.95, value=0.25, step=0.01, label="Score threshold")
                    mask_threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Mask threshold")
                    crop_padding = gr.Slider(0, 64, value=8, step=1, label="Crop padding")

            with gr.Column(scale=2):
                status = gr.Textbox(label="Status", interactive=False)
                overlay = gr.Image(type="pil", label="Detected line masks")
                crops = gr.Gallery(
                    label="Line crops",
                    columns=2,
                    height=520,
                    object_fit="contain",
                )

        run.click(
            fn=detector.detect,
            inputs=[image, image_size, threshold, mask_threshold, load_max_side, crop_padding],
            outputs=[overlay, crops, status],
        )
        image.change(
            fn=detector.detect,
            inputs=[image, image_size, threshold, mask_threshold, load_max_side, crop_padding],
            outputs=[overlay, crops, status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for Mask2Former text line masks.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    demo = build_demo(model_dir=args.model_dir, cpu=args.cpu)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
