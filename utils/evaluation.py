from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from .data import labels_to_device
from .inference import predict_masks
from .postprocess import mask_iou, mask_to_contour, postprocess_line_masks


@torch.no_grad()
def evaluate_loss(model, loader, device, use_amp, max_steps=None):
    model.eval()
    total_loss = 0.0
    steps = 0
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for batch in tqdm(loader, desc="val", leave=True, dynamic_ncols=True):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device) if "pixel_mask" in batch else None
        mask_labels = labels_to_device(batch["mask_labels"], device)
        class_labels = labels_to_device(batch["class_labels"], device)
        with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=use_amp):
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
        total_loss += float(outputs.loss.detach().cpu())
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return total_loss / max(1, steps)


def polygon_masks(polygons, image_size):
    width, height = image_size
    masks = []
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        mask_image = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask_image).polygon(polygon, fill=1)
        mask = np.asarray(mask_image, dtype=np.uint8)
        if mask.sum() > 0:
            masks.append(mask.astype(bool))
    return masks


def match_masks(gt_masks, pred_masks, iou_threshold):
    matches = 0
    used_pred = set()
    for gt_mask in gt_masks:
        best_index = None
        best_iou = 0.0
        for index, pred_mask in enumerate(pred_masks):
            if index in used_pred:
                continue
            iou = mask_iou(gt_mask, pred_mask)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index is not None and best_iou >= iou_threshold:
            matches += 1
            used_pred.add(best_index)
    return matches


@torch.no_grad()
def evaluate_detection_metrics(
    model,
    processor,
    dataset,
    device,
    image_size,
    threshold,
    mask_threshold,
    iou_threshold,
    max_images,
):
    model.eval()
    tp = 0
    total_gt = 0
    total_pred = 0
    if max_images <= 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "gt": 0,
            "pred": 0,
            "skipped": True,
        }
    images = min(max_images, len(dataset))
    indices = random.Random(1701).sample(range(len(dataset)), images)
    progress = tqdm(indices, desc="metrics", leave=True, dynamic_ncols=True)
    for index in progress:
        item = dataset[index]
        image = item["image"]
        try:
            pred_masks, pred_scores = predict_masks(model, processor, image, device, image_size, threshold, mask_threshold)
            detections = postprocess_line_masks(pred_masks, pred_scores, min_score=threshold)
            pred_only = [mask for mask, _ in detections]
            gt_masks = polygon_masks(item["polygons"], image.size)
            tp += match_masks(gt_masks, pred_only, iou_threshold)
            total_gt += len(gt_masks)
            total_pred += len(pred_only)
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"metrics skipped for {item['path']}: {type(exc).__name__}: {exc}")
        precision = tp / total_pred if total_pred else 0.0
        recall = tp / total_gt if total_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        progress.set_postfix(p=f"{precision:.3f}", r=f"{recall:.3f}", f1=f"{f1:.3f}")

    precision = tp / total_pred if total_pred else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "gt": total_gt,
        "pred": total_pred,
    }


def draw_preview(image, gt_polygons, pred_masks, pred_scores, output_path, min_score=0.25):
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for polygon in gt_polygons:
        draw.line(polygon + [polygon[0]], fill=(40, 220, 80, 230), width=2)

    for mask, score in postprocess_line_masks(pred_masks, pred_scores, min_score=min_score):
        contour = mask_to_contour(mask)
        if contour is None:
            continue
        points = [(float(x), float(y)) for x, y in contour]
        draw.polygon(points, fill=(255, 80, 80, 45))
        draw.line(points + [points[0]], fill=(255, 80, 80, 230), width=2)
        x_min = min(x for x, _ in points)
        y_min = min(y for _, y in points)
        draw.rectangle([x_min, y_min, x_min + 60, y_min + 18], fill=(255, 80, 80, 190))
        draw.text((x_min + 4, y_min + 1), f"{float(score):.2f}", fill=(0, 0, 0, 255))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def write_epoch_previews(
    model,
    processor,
    dataset,
    output_dir,
    device,
    epoch,
    image_size,
    threshold,
    mask_threshold,
    count,
):
    if count <= 0:
        return
    indices = random.Random(1009 + epoch).sample(range(len(dataset)), min(count, len(dataset)))
    for index in indices:
        item = dataset[index]
        image = item["image"]
        try:
            pred_masks, pred_scores = predict_masks(model, processor, image, device, image_size, threshold, mask_threshold)
            output_path = Path(output_dir) / f"epoch_{epoch:03d}_{Path(item['path']).stem}.png"
            draw_preview(
                image,
                item["polygons"],
                pred_masks,
                pred_scores,
                output_path,
                min_score=threshold,
            )
            print(f"preview: {output_path} gt={len(item['polygons'])} pred={len(pred_scores)}")
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"preview skipped for {item['path']}: {type(exc).__name__}: {exc}")
