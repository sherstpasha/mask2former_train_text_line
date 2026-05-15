import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.training import TrainConfig, run_training


def parse_args():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train Mask2Former line instance masks from scratch, or resume a local checkpoint."
    )
    parser.add_argument(
        "--model-config",
        "--model-name",
        dest="model_config",
        default=defaults.model_config,
        help="HuggingFace config/image-processor source. Model weights are initialized from scratch.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        "--resume-training-state",
        dest="resume_from_checkpoint",
        default=defaults.resume_from_checkpoint,
        help="Local checkpoint directory, for example mask2former_lines_scratch/latest or step_checkpoint.",
    )
    parser.add_argument("--auto-resume", action="store_true", default=defaults.auto_resume)
    parser.add_argument("--train-img-dir", default=defaults.train_img_dir)
    parser.add_argument("--train-gt-dir", default=defaults.train_gt_dir)
    parser.add_argument("--val-img-dir", default=defaults.val_img_dir)
    parser.add_argument("--val-gt-dir", default=defaults.val_gt_dir)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--load-max-side", type=int, default=defaults.load_max_side)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--accumulation-steps", type=int, default=defaults.accumulation_steps)
    parser.add_argument("--max-instances-per-image", type=int, default=defaults.max_instances_per_image)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-train-steps", type=int, default=defaults.max_train_steps)
    parser.add_argument("--max-val-steps", type=int, default=defaults.max_val_steps)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=defaults.checkpoint_interval_steps)
    parser.add_argument("--preview-count", type=int, default=defaults.preview_count)
    parser.add_argument("--preview-threshold", type=float, default=defaults.preview_threshold)
    parser.add_argument("--preview-mask-threshold", type=float, default=defaults.preview_mask_threshold)
    parser.add_argument("--metric-images", type=int, default=defaults.metric_images)
    parser.add_argument("--metric-iou-threshold", type=float, default=defaults.metric_iou_threshold)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=defaults.gradient_checkpointing)
    parser.add_argument("--augment", dest="augment", action="store_true", default=defaults.augment)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument(
        "--augmentation-strength",
        choices=["light", "medium", "strong"],
        default=defaults.augmentation_strength,
    )
    parser.add_argument("--no-amp", action="store_true", default=defaults.no_amp)
    parser.add_argument("--cpu", action="store_true", default=defaults.cpu)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_training(TrainConfig(**vars(args)))
    print(result)


if __name__ == "__main__":
    main()
