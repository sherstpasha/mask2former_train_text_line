import numpy as np
import onnxruntime as ort


IMAGENET_MEAN = np.array([0.48500001430511475, 0.4560000002384186, 0.4059999883174896], dtype=np.float32)
IMAGENET_STD = np.array([0.2290000021457672, 0.2239999920129776, 0.22499999403953552], dtype=np.float32)


def sigmoid(x):
    result = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def resize_bilinear_chw(values, height, width):
    channels, in_height, in_width = values.shape
    if (in_height, in_width) == (height, width):
        return values.copy()

    y = (np.arange(height, dtype=np.float32) + 0.5) * in_height / height - 0.5
    x = (np.arange(width, dtype=np.float32) + 0.5) * in_width / width - 0.5
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = y - y0
    wx = x - x0

    y0 = np.clip(y0, 0, in_height - 1)
    y1 = np.clip(y1, 0, in_height - 1)
    x0 = np.clip(x0, 0, in_width - 1)
    x1 = np.clip(x1, 0, in_width - 1)

    top = values[:, y0][:, :, x0] * (1.0 - wx) + values[:, y0][:, :, x1] * wx
    bottom = values[:, y1][:, :, x0] * (1.0 - wx) + values[:, y1][:, :, x1] * wx
    return (top * (1.0 - wy)[None, :, None] + bottom * wy[None, :, None]).astype(np.float32)


def resize_nearest_hw(mask, height, width):
    in_height, in_width = mask.shape
    if (in_height, in_width) == (height, width):
        return mask.copy()
    y = np.floor(np.arange(height, dtype=np.float32) * in_height / height).astype(np.int64)
    x = np.floor(np.arange(width, dtype=np.float32) * in_width / width).astype(np.int64)
    return mask[np.clip(y, 0, in_height - 1)[:, None], np.clip(x, 0, in_width - 1)[None, :]]


def preprocess_rgb(image_rgb, image_size=512):
    image = np.asarray(image_rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")

    chw = np.transpose(image, (2, 0, 1))
    chw = resize_bilinear_chw(chw, image_size, image_size)
    chw = np.transpose(chw, (1, 2, 0)) / 255.0
    chw = (chw - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(chw, (2, 0, 1))[None].astype(np.float32)
    pixel_mask = np.ones((1, image_size, image_size), dtype=np.int64)
    return {"pixel_values": chw, "pixel_mask": pixel_mask}


def mask_area(mask):
    return int(mask.sum())


def mask_iou(first, second):
    intersection = np.logical_and(first, second).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def postprocess_line_masks(binary_masks, scores, min_score=0.25, min_area=16, nms_iou=0.5):
    candidates = []
    for mask, score in zip(binary_masks, scores):
        score = float(score)
        if score < min_score:
            continue
        mask = mask.astype(bool)
        if mask_area(mask) < min_area:
            continue
        candidates.append((mask, score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    kept = []
    for mask, score in candidates:
        if all(mask_iou(mask, prev_mask) <= nms_iou for prev_mask, _ in kept):
            kept.append((mask, score))

    kept.sort(key=lambda item: ((mask_bbox(item[0])[1] + mask_bbox(item[0])[3]) / 2.0, mask_bbox(item[0])[0]))
    return kept


def postprocess_instance_segmentation(class_logits, mask_logits, target_size, threshold=0.25):
    class_logits = class_logits[0]
    mask_logits = mask_logits[0]
    num_queries = class_logits.shape[0]
    num_classes = class_logits.shape[-1] - 1

    scores = softmax(class_logits, axis=-1)[:, :-1]
    flat_indices = np.argpartition(scores.reshape(-1), -num_queries)[-num_queries:]
    scores_per_image = scores.reshape(-1)[flat_indices]
    query_indices = flat_indices // num_classes
    order = np.argsort(-scores_per_image)
    scores_per_image = scores_per_image[order]
    query_indices = query_indices[order]

    selected_logits = mask_logits[query_indices]
    selected_logits = resize_bilinear_chw(selected_logits, 384, 384)
    pred_masks_384 = selected_logits > 0
    mask_probs = sigmoid(selected_logits)
    mask_scores = (mask_probs.reshape(num_queries, -1) * pred_masks_384.reshape(num_queries, -1)).sum(axis=1)
    mask_scores /= pred_masks_384.reshape(num_queries, -1).sum(axis=1) + 1e-6
    pred_scores = scores_per_image * mask_scores

    target_height, target_width = target_size
    masks = []
    scores_out = []
    for mask_384, score in zip(pred_masks_384, pred_scores):
        if score < threshold or not mask_384.any():
            continue
        masks.append(resize_nearest_hw(mask_384, target_height, target_width))
        scores_out.append(float(score))
    return masks, scores_out


class Mask2FormerOnnxLineDetector:
    def __init__(self, model_path, image_size=512, providers=None):
        self.image_size = int(image_size)
        self.session = ort.InferenceSession(
            str(model_path),
            providers=providers or ["CPUExecutionProvider"],
        )

    def predict(self, image_rgb, threshold=0.25, min_area=16, nms_iou=0.5):
        image = np.asarray(image_rgb)
        target_size = image.shape[:2]
        feeds = preprocess_rgb(image, self.image_size)
        class_logits, mask_logits = self.session.run(None, feeds)
        masks, scores = postprocess_instance_segmentation(
            class_logits,
            mask_logits,
            target_size=target_size,
            threshold=threshold,
        )
        return postprocess_line_masks(masks, scores, min_score=threshold, min_area=min_area, nms_iou=nms_iou)
