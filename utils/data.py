import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import ConcatDataset, Dataset

try:
    import cv2
except ImportError:
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BILINEAR = getattr(Image, "Resampling", Image).BILINEAR

AUGMENTATION_PROFILES = {
    "light": {
        "probability": 0.65,
        "geometry_probability": 0.35,
        "color_probability": 0.55,
        "noise_probability": 0.12,
        "blur_probability": 0.08,
        "jpeg_probability": 0.08,
        "grayscale_probability": 0.04,
        "rotate_degrees": 2.0,
        "translate_fraction": 0.02,
        "scale_delta": 0.04,
        "shear_degrees": 1.0,
        "brightness_range": (0.85, 1.15),
        "contrast_range": (0.85, 1.18),
        "color_range": (0.85, 1.12),
        "noise_sigma_range": (3.0, 8.0),
        "blur_radius_range": (0.2, 0.7),
        "jpeg_quality_range": (65, 90),
    },
    "medium": {
        "probability": 0.85,
        "geometry_probability": 0.55,
        "color_probability": 0.75,
        "noise_probability": 0.22,
        "blur_probability": 0.16,
        "jpeg_probability": 0.14,
        "grayscale_probability": 0.08,
        "rotate_degrees": 7.0,
        "translate_fraction": 0.07,
        "scale_delta": 0.12,
        "shear_degrees": 4.0,
        "brightness_range": (0.70, 1.30),
        "contrast_range": (0.70, 1.40),
        "color_range": (0.65, 1.30),
        "noise_sigma_range": (4.0, 16.0),
        "blur_radius_range": (0.4, 1.5),
        "jpeg_quality_range": (35, 80),
    },
    "strong": {
        "probability": 0.95,
        "geometry_probability": 0.70,
        "color_probability": 0.85,
        "noise_probability": 0.30,
        "blur_probability": 0.22,
        "jpeg_probability": 0.20,
        "grayscale_probability": 0.12,
        "rotate_degrees": 12.0,
        "translate_fraction": 0.12,
        "scale_delta": 0.20,
        "shear_degrees": 7.0,
        "brightness_range": (0.50, 1.50),
        "contrast_range": (0.50, 1.70),
        "color_range": (0.40, 1.60),
        "noise_sigma_range": (8.0, 28.0),
        "blur_radius_range": (0.8, 2.5),
        "jpeg_quality_range": (18, 65),
    },
}


def parse_gt_line(line):
    parts = line.strip().lstrip("\ufeff").split(",")
    coords = []
    for value in parts:
        try:
            coords.append(float(value))
        except ValueError:
            break
    if len(coords) < 6:
        return None
    if len(coords) % 2 == 1:
        coords = coords[:-1]
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]


def read_polygons(gt_path):
    polygons = []
    if not gt_path.exists():
        return polygons
    with gt_path.open("r", encoding="utf-8") as file:
        for line in file:
            polygon = parse_gt_line(line)
            if polygon:
                polygons.append(polygon)
    return polygons


def resize_image_and_polygons(image, polygons, max_side):
    if not max_side:
        return image, polygons
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_side:
        return image, polygons
    scale = max_side / long_side
    new_size = (int(round(width * scale)), int(round(height * scale)))
    image = image.resize(new_size, BILINEAR)
    scaled = [[(x * scale, y * scale) for x, y in polygon] for polygon in polygons]
    return image, scaled


def polygons_to_instance_map(polygons, image_size):
    width, height = image_size
    instance_image = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(instance_image)
    instance_id_to_semantic_id = {}
    for index, polygon in enumerate(polygons, start=1):
        if len(polygon) < 3:
            continue
        draw.polygon(polygon, fill=index)
        instance_id_to_semantic_id[index] = 0
    return np.asarray(instance_image, dtype=np.int32), instance_id_to_semantic_id


def resize_polygons(polygons, from_size, to_size):
    from_width, from_height = from_size
    to_width, to_height = to_size
    scale_x = to_width / from_width
    scale_y = to_height / from_height
    return [[(x * scale_x, y * scale_y) for x, y in polygon] for polygon in polygons]


def letterbox_image(image, image_size, fill=(255, 255, 255)):
    """Resize without changing aspect ratio and center on a square canvas."""
    width, height = image.size
    scale = min(image_size / width, image_size / height)
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resized = image.resize(resized_size, BILINEAR)
    left = (image_size - resized_size[0]) // 2
    top = (image_size - resized_size[1]) // 2
    canvas = Image.new("RGB", (image_size, image_size), fill)
    canvas.paste(resized, (left, top))
    return canvas, scale, left, top, resized_size


def polygons_to_binary_masks(polygons, image_size):
    """Keep instances independent so overlapping line polygons are not overwritten."""
    width, height = image_size
    masks = []
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        mask_image = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask_image).polygon(polygon, fill=1)
        mask = np.asarray(mask_image, dtype=np.uint8)
        if mask.any():
            masks.append(mask)
    if not masks:
        return np.zeros((0, height, width), dtype=np.uint8)
    return np.stack(masks, axis=0)


def resize_sample_to_model_size(image, polygons, image_size):
    image, scale, left, top, _ = letterbox_image(image, image_size)
    polygons = [
        [(x * scale + left, y * scale + top) for x, y in polygon]
        for polygon in polygons
    ]
    return image, polygons_to_binary_masks(polygons, image.size)


def affine_augmentation_matrix(width, height, profile):
    center_x = width / 2.0
    center_y = height / 2.0
    angle = random.uniform(-profile["rotate_degrees"], profile["rotate_degrees"])
    scale_delta = profile["scale_delta"]
    scale = random.uniform(1.0 - scale_delta, 1.0 + scale_delta)
    shear = np.deg2rad(random.uniform(-profile["shear_degrees"], profile["shear_degrees"]))
    translate = profile["translate_fraction"]
    translate_x = random.uniform(-translate, translate) * width
    translate_y = random.uniform(-translate, translate) * height

    to_origin = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]], dtype=np.float32)
    rotation = np.array(
        [
            [np.cos(np.deg2rad(angle)) * scale, -np.sin(np.deg2rad(angle)) * scale, 0],
            [np.sin(np.deg2rad(angle)) * scale, np.cos(np.deg2rad(angle)) * scale, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    shear_matrix = np.array([[1, np.tan(shear), 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    back = np.array([[1, 0, center_x + translate_x], [0, 1, center_y + translate_y], [0, 0, 1]], dtype=np.float32)
    return (back @ shear_matrix @ rotation @ to_origin)[:2]


def apply_geometric_augmentation(image, instance_masks, profile):
    if cv2 is None:
        raise ImportError("opencv-python is required for geometric training augmentations")

    width, height = image.size
    matrix = affine_augmentation_matrix(width, height, profile)
    image_array = np.asarray(image.convert("RGB"))
    augmented_image = cv2.warpAffine(
        image_array,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    augmented_masks = []
    for mask in instance_masks:
        augmented_masks.append(
            cv2.warpAffine(
                mask,
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(np.uint8, copy=False)
        )
    if augmented_masks:
        augmented_masks = np.stack(augmented_masks, axis=0)
    else:
        augmented_masks = np.zeros((0, height, width), dtype=np.uint8)
    return Image.fromarray(augmented_image, mode="RGB"), augmented_masks


def apply_color_augmentation(image, profile):
    if random.random() < profile["color_probability"]:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(*profile["brightness_range"]))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(*profile["contrast_range"]))
        image = ImageEnhance.Color(image).enhance(random.uniform(*profile["color_range"]))
    if random.random() < profile["grayscale_probability"]:
        image = ImageOps.grayscale(image).convert("RGB")
    if random.random() < profile["blur_probability"]:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(*profile["blur_radius_range"])))
    if random.random() < profile["noise_probability"]:
        array = np.asarray(image).astype(np.float32)
        sigma = random.uniform(*profile["noise_sigma_range"])
        noise = np.random.normal(0.0, sigma, array.shape).astype(np.float32)
        image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")
    if random.random() < profile["jpeg_probability"]:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(*profile["jpeg_quality_range"]))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    return image


def augment_image_and_segmentation(image, segmentation_map, strength="medium"):
    profile = AUGMENTATION_PROFILES.get(strength)
    if profile is None:
        raise ValueError(f"unknown augmentation strength: {strength}")
    if random.random() >= profile["probability"]:
        return image, segmentation_map
    if random.random() < profile["geometry_probability"]:
        image, segmentation_map = apply_geometric_augmentation(image, segmentation_map, profile)
    image = apply_color_augmentation(image, profile)
    return image, segmentation_map


class LineInstanceDataset(Dataset):
    def __init__(self, img_dir, gt_dir, load_max_side=1024):
        self.img_dir = Path(img_dir)
        self.gt_dir = Path(gt_dir)
        self.load_max_side = load_max_side
        self.images = sorted(
            path for path in self.img_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.images:
            raise FileNotFoundError(f"no images found in {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        gt_path = self.gt_dir / f"gt_{image_path.stem}.txt"
        polygons = read_polygons(gt_path)
        image, polygons = resize_image_and_polygons(image, polygons, self.load_max_side)
        return {
            "image": image,
            "path": str(image_path),
            "polygons": polygons,
        }


def encode_instance_targets(segmentation_maps, max_instances_per_image):
    mask_labels = []
    class_labels = []
    for instance_masks in segmentation_maps:
        masks = torch.from_numpy(instance_masks.astype(np.float32, copy=False))
        if max_instances_per_image and len(masks) > max_instances_per_image:
            areas = masks.sum(dim=(1, 2))
            keep = torch.argsort(areas, descending=True)[:max_instances_per_image]
            masks = masks[keep]
        classes = torch.zeros((len(masks),), dtype=torch.int64)
        mask_labels.append(masks)
        class_labels.append(classes)
    return mask_labels, class_labels


def collate_fn(processor, image_size, max_instances_per_image, augment=False, augmentation_strength="medium"):
    def collate(batch):
        images = []
        segmentation_maps = []
        for item in batch:
            image, segmentation_map = resize_sample_to_model_size(
                item["image"],
                item["polygons"],
                image_size,
            )
            if augment:
                image, segmentation_map = augment_image_and_segmentation(
                    image,
                    segmentation_map,
                    strength=augmentation_strength,
                )
            images.append(image)
            segmentation_maps.append(segmentation_map)
        encoded = processor(
            images=images,
            do_resize=False,
            return_tensors="pt",
        )
        mask_labels, class_labels = encode_instance_targets(segmentation_maps, max_instances_per_image)
        encoded["mask_labels"] = mask_labels
        encoded["class_labels"] = class_labels
        return encoded

    return collate


def split_paths(value):
    return [item.strip() for item in str(value).split(";") if item.strip()]


def build_dataset(img_dirs, gt_dirs, load_max_side):
    img_paths = split_paths(img_dirs)
    gt_paths = split_paths(gt_dirs)
    if len(img_paths) != len(gt_paths):
        raise ValueError(f"image dirs and gt dirs count mismatch: {len(img_paths)} != {len(gt_paths)}")
    datasets = [LineInstanceDataset(img_dir, gt_dir, load_max_side) for img_dir, gt_dir in zip(img_paths, gt_paths)]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def labels_to_device(labels, device):
    return [label.to(device) for label in labels]
