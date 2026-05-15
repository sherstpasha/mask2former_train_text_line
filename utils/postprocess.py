import cv2
import numpy as np
import torch


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
		return [0, 0, 0, 0]
	return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def postprocess_line_masks(binary_masks, scores, min_score=0.25, min_area=16, nms_iou=0.5):
	if binary_masks is None:
		return []
	if torch.is_tensor(binary_masks):
		binary_masks = binary_masks.detach().cpu().numpy()
	if torch.is_tensor(scores):
		scores = scores.detach().cpu().tolist()

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


def mask_to_contour(mask):
	mask_u8 = mask.astype(np.uint8) * 255
	contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return None
	contour = max(contours, key=cv2.contourArea)
	if len(contour) < 3:
		return None
	return contour[:, 0, :]
