"""Run Mask2Former ONNX inference in Gradio with selectable CPU/CUDA execution."""

import argparse
import sys
from pathlib import Path

import gradio as gr
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.onnx_inference import (
    Mask2FormerOnnxDetector,
    available_devices,
    crop_detection,
    draw_detections,
)


DEFAULT_MODEL_DIR = ROOT / "mask2former_lines_swin_small_page_medium_ft100" / "best_f1"


def build_demo(model_dir):
    model_dir = Path(model_dir).expanduser().resolve()
    detector = Mask2FormerOnnxDetector(
        fp32_model=model_dir / "model.onnx",
        fp16_model=model_dir / "model.fp16.onnx",
    )
    devices = available_devices()

    def detect(image, device, threshold, nms_iou, crop_padding):
        if image is None:
            return None, [], "Загрузите изображение."
        source = Image.fromarray(image).convert("RGB")
        detections = detector.predict(source, device=device, threshold=float(threshold), nms_iou=float(nms_iou))
        overlay = draw_detections(source, detections)
        crops = [
            (crop_detection(source, mask, int(crop_padding)), f"line {index}: {score:.2f}")
            for index, (mask, score) in enumerate(detections, start=1)
        ]
        precision = "FP16" if device == "CUDA" and detector.fp16_model.exists() else "FP32"
        return overlay, crops, f"Найдено строк: {len(detections)} · {device} · {precision}"

    with gr.Blocks(title="Mask2Former ONNX text lines") as demo:
        gr.Markdown("# Mask2Former ONNX: сегментация строк текста")
        gr.Markdown(f"Модель: `{model_dir}` · доступно: `{', '.join(devices)}`")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="numpy", label="Изображение")
                device = gr.Radio(devices, value=devices[0], label="Устройство")
                run = gr.Button("Найти строки", variant="primary")
                with gr.Accordion("Параметры", open=False):
                    threshold = gr.Slider(0.01, 0.95, value=0.15, step=0.01, label="Score threshold")
                    nms_iou = gr.Slider(0.05, 0.95, value=0.50, step=0.05, label="NMS IoU")
                    crop_padding = gr.Slider(0, 64, value=8, step=1, label="Отступ crop")
            with gr.Column(scale=2):
                status = gr.Textbox(label="Статус", interactive=False)
                overlay = gr.Image(type="pil", label="Найденные строки")
                crops = gr.Gallery(label="Вырезанные строки", columns=2, height=520, object_fit="contain")

        run.click(
            detect,
            inputs=[image, device, threshold, nms_iou, crop_padding],
            outputs=[overlay, crops, status],
        )
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    build_demo(args.model_dir).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
