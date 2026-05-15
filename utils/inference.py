import numpy as np
import torch
from PIL import Image, ImageDraw

from .postprocess import mask_bbox, mask_to_contour, postprocess_line_masks


BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def resize_for_inference(image, max_side):
    if not max_side:
        return image
    width, height = image.size
    if max(width, height) <= max_side:
        return image
    scale = max_side / max(width, height)
    return image.resize((int(round(width * scale)), int(round(height * scale))), BILINEAR)


@torch.no_grad()
def predict_masks(model, processor, image, device, image_size, threshold, mask_threshold):
    model.eval()
    inputs = processor(images=image, size={"height": image_size, "width": image_size}, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=[image.size[::-1]],
        return_binary_maps=True,
    )[0]
    if results["segmentation"] is None:
        return [], []
    scores = [item["score"] for item in results["segments_info"]]
    return results["segmentation"], scores


def detect_image(model, processor, image, device, image_size, threshold, mask_threshold):
    pred_masks, pred_scores = predict_masks(
        model,
        processor,
        image,
        device,
        image_size,
        threshold,
        mask_threshold,
    )
    return postprocess_line_masks(pred_masks, pred_scores, min_score=threshold)


def draw_masks(image, detections, numbered=False):
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for index, (mask, score) in enumerate(detections, start=1):
        contour = mask_to_contour(mask)
        if contour is None:
            continue
        points = [(float(x), float(y)) for x, y in contour]
        draw.polygon(points, fill=(40, 220, 80, 55))
        draw.line(points + [points[0]], fill=(40, 220, 80, 230), width=2)
        x_min = min(x for x, _ in points)
        y_min = min(y for _, y in points)
        label = f"{index}: {score:.2f}" if numbered else f"{float(score):.2f}"
        draw.rectangle([x_min, y_min, x_min + max(60, 10 + 8 * len(label)), y_min + 18], fill=(40, 220, 80, 190))
        draw.text((x_min + 4, y_min + 1), label, fill=(0, 0, 0, 255))
    return overlay


def crop_line(image, mask, padding):
    x_min, y_min, x_max, y_max = mask_bbox(mask)
    width, height = image.size
    left = max(0, int(np.floor(x_min)) - padding)
    top = max(0, int(np.floor(y_min)) - padding)
    right = min(width, int(np.ceil(x_max)) + padding + 1)
    bottom = min(height, int(np.ceil(y_max)) + padding + 1)

    crop = image.crop((left, top, right, bottom)).convert("RGBA")
    crop_mask = Image.fromarray((mask[top:bottom, left:right].astype(np.uint8) * 255), mode="L")

    white = Image.new("RGBA", crop.size, (255, 255, 255, 255))
    return Image.composite(crop, white, crop_mask).convert("RGB")
