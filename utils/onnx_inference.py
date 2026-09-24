"""Dependency-light Mask2Former inference using ONNX Runtime, NumPy and Pillow."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
NEAREST = getattr(Image, "Resampling", Image).NEAREST


def available_devices():
    providers = ort.get_available_providers()
    devices = ["CPU"]
    if "CUDAExecutionProvider" in providers:
        devices.insert(0, "CUDA")
    return devices


def letterbox(image, image_size):
    width, height = image.size
    scale = min(image_size / width, image_size / height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(resized_size, BILINEAR)
    left = (image_size - resized_size[0]) // 2
    top = (image_size - resized_size[1]) // 2
    canvas = Image.new("RGB", (image_size, image_size), "white")
    canvas.paste(resized, (left, top))
    return canvas, left, top, resized_size


def resize_mask(mask, size, resample):
    image = Image.fromarray(np.asarray(mask, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, resample), dtype=np.float32)


def softmax(values, axis=-1):
    values = values - values.max(axis=axis, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=axis, keepdims=True)


def sigmoid(values):
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def mask_iou(first, second):
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


class Mask2FormerOnnxDetector:
    def __init__(self, fp32_model, fp16_model=None, image_size=1024):
        self.fp32_model = Path(fp32_model).expanduser().resolve()
        self.fp16_model = Path(fp16_model).expanduser().resolve() if fp16_model else None
        self.image_size = int(image_size)
        self._sessions = {}

    def _session(self, device):
        device = str(device).upper()
        if device not in available_devices():
            raise RuntimeError(f"{device} is unavailable; installed providers: {ort.get_available_providers()}")
        if device not in self._sessions:
            use_fp16 = device == "CUDA" and self.fp16_model and self.fp16_model.exists()
            model_path = self.fp16_model if use_fp16 else self.fp32_model
            provider = "CUDAExecutionProvider" if device == "CUDA" else "CPUExecutionProvider"
            self._sessions[device] = ort.InferenceSession(str(model_path), providers=[provider])
        return self._sessions[device]

    def predict(self, image, device="CPU", threshold=0.15, nms_iou=0.5):
        image = image.convert("RGB")
        original_size = image.size
        canvas, left, top, resized_size = letterbox(image, self.image_size)
        pixels = np.asarray(canvas, dtype=np.float32) / 255.0
        pixels = ((pixels - MEAN) / STD).transpose(2, 0, 1)[None]

        session = self._session(device)
        input_type = session.get_inputs()[0].type
        pixels = pixels.astype(np.float16 if input_type == "tensor(float16)" else np.float32)
        pixel_mask = np.ones((1, self.image_size, self.image_size), dtype=np.int64)
        class_logits, mask_logits = session.run(
            None,
            {"pixel_values": pixels, "pixel_mask": pixel_mask},
        )

        class_scores = softmax(class_logits[0].astype(np.float32), axis=-1)[:, :-1].max(axis=-1)
        candidates = []
        for query, class_score in enumerate(class_scores):
            logits_384 = resize_mask(mask_logits[0, query], (384, 384), BILINEAR)
            binary_384 = logits_384 > 0.0
            if not binary_384.any():
                continue
            mask_score = float(sigmoid(logits_384)[binary_384].mean())
            score = float(class_score * mask_score)
            if score < threshold:
                continue

            mask_1024 = resize_mask(binary_384.astype(np.float32), (self.image_size, self.image_size), NEAREST) > 0.5
            cropped = mask_1024[top : top + resized_size[1], left : left + resized_size[0]]
            restored = resize_mask(cropped.astype(np.float32), original_size, NEAREST) > 0.5
            if restored.sum() >= 16:
                candidates.append((restored, score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        kept = []
        for mask, score in candidates:
            if all(mask_iou(mask, old_mask) <= nms_iou for old_mask, _ in kept):
                kept.append((mask, score))
        kept.sort(key=lambda item: ((mask_bbox(item[0])[1] + mask_bbox(item[0])[3]) / 2, mask_bbox(item[0])[0]))
        return kept


def draw_detections(image, detections):
    result = image.convert("RGBA")
    for index, (mask, score) in enumerate(detections, start=1):
        color = Image.new("RGBA", result.size, (40, 220, 80, 0))
        color.putalpha(Image.fromarray(mask.astype(np.uint8) * 60, mode="L"))
        result = Image.alpha_composite(result, color)
        draw = ImageDraw.Draw(result)
        x1, y1, x2, y2 = mask_bbox(mask)
        draw.rectangle((x1, y1, x2, y2), outline=(40, 220, 80, 255), width=2)
        draw.text((x1 + 3, y1 + 2), f"{index}: {score:.2f}", fill=(0, 0, 0, 255), stroke_width=2, stroke_fill="white")
    return result.convert("RGB")


def crop_detection(image, mask, padding=8):
    x1, y1, x2, y2 = mask_bbox(mask)
    left = max(0, x1 - padding)
    top = max(0, y1 - padding)
    right = min(image.width, x2 + padding + 1)
    bottom = min(image.height, y2 + padding + 1)
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    crop_mask = Image.fromarray(mask[top:bottom, left:right].astype(np.uint8) * 255, mode="L")
    return Image.composite(crop, Image.new("RGB", crop.size, "white"), crop_mask)
