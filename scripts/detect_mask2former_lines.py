import argparse
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data import IMAGE_EXTENSIONS
from utils.inference import detect_image, draw_masks, resize_for_inference


def main():
    parser = argparse.ArgumentParser(description="Draw Mask2Former line instance masks.")
    parser.add_argument("--model-dir", default="mask2former_lines_scratch/best")
    parser.add_argument("--img-dir", default="handwritten_essay_v2_east_lines/test_img")
    parser.add_argument("--output-dir", default="mask2former_lines_preview")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--load-max-side", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForUniversalSegmentation.from_pretrained(args.model_dir).to(device)
    model.eval()

    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in img_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"no images found in {img_dir}")

    for image_path in random.sample(images, min(args.count, len(images))):
        image = Image.open(image_path).convert("RGB")
        image = resize_for_inference(image, args.load_max_side)
        detections = detect_image(
            model,
            processor,
            image,
            device,
            args.image_size,
            args.threshold,
            args.mask_threshold,
        )
        overlay = draw_masks(image, detections)
        output_path = output_dir / f"{image_path.stem}_mask2former.png"
        overlay.save(output_path)
        print(f"{output_path} masks={len(detections)}")


if __name__ == "__main__":
    main()
