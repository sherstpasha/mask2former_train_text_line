import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.training import TrainConfig, run_training


SCHOOL_ROOT = "school_notebooks_RU_lines"
IAM_ROOT = "coco_lines_iam_test"

SCHOOL_TRAIN_SPLITS = ["train"]
SCHOOL_VAL_SPLITS = ["val"]
IAM_TRAIN_SPLITS = ["test"]
IAM_VAL_SPLITS = []

CONFIG = {
    "model_config": "facebook/mask2former-swin-tiny-coco-instance",
    "resume_from_checkpoint": None,
    "auto_resume": True,
    "output_dir": "mask2former_lines_school_iam_scratch",
    "image_size": 1024,
    "load_max_side": 1024,
    "epochs": 50,
    "batch_size": 1,
    "accumulation_steps": 8,
    "max_instances_per_image": 32,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "seed": 42,
    "max_train_steps": None,
    "max_val_steps": None,
    "checkpoint_interval_steps": 100,
    "preview_count": 3,
    "preview_threshold": 0.25,
    "preview_mask_threshold": 0.5,
    "metric_images": 20,
    "metric_iou_threshold": 0.5,
    "gradient_checkpointing": False,
    "augment": True,
    "augmentation_strength": "medium",
    "no_amp": False,
    "cpu": False,
}


def collect_split_dirs(root, splits):
    root = Path(root)
    img_dirs = []
    gt_dirs = []
    for split in splits:
        img_dir = root / f"{split}_img"
        gt_dir = root / f"{split}_gt"
        if not img_dir.exists():
            raise FileNotFoundError(f"missing image directory: {img_dir}")
        if not gt_dir.exists():
            raise FileNotFoundError(f"missing gt directory: {gt_dir}")
        img_dirs.append(str(img_dir))
        gt_dirs.append(str(gt_dir))
    return img_dirs, gt_dirs


def join_paths(paths):
    return ";".join(str(path) for path in paths)


def build_config():
    train_img_dirs = []
    train_gt_dirs = []
    val_img_dirs = []
    val_gt_dirs = []

    img_dirs, gt_dirs = collect_split_dirs(SCHOOL_ROOT, SCHOOL_TRAIN_SPLITS)
    train_img_dirs.extend(img_dirs)
    train_gt_dirs.extend(gt_dirs)

    img_dirs, gt_dirs = collect_split_dirs(IAM_ROOT, IAM_TRAIN_SPLITS)
    train_img_dirs.extend(img_dirs)
    train_gt_dirs.extend(gt_dirs)

    img_dirs, gt_dirs = collect_split_dirs(SCHOOL_ROOT, SCHOOL_VAL_SPLITS)
    val_img_dirs.extend(img_dirs)
    val_gt_dirs.extend(gt_dirs)

    if IAM_VAL_SPLITS:
        img_dirs, gt_dirs = collect_split_dirs(IAM_ROOT, IAM_VAL_SPLITS)
        val_img_dirs.extend(img_dirs)
        val_gt_dirs.extend(gt_dirs)

    config_values = dict(CONFIG)
    config_values.update(
        {
            "train_img_dir": join_paths(train_img_dirs),
            "train_gt_dir": join_paths(train_gt_dirs),
            "val_img_dir": join_paths(val_img_dirs),
            "val_gt_dir": join_paths(val_gt_dirs),
        }
    )
    return TrainConfig(**config_values)


def main():
    config = build_config()
    print("train image dirs:", config.train_img_dir)
    print("train gt dirs:", config.train_gt_dir)
    print("val image dirs:", config.val_img_dir)
    print("val gt dirs:", config.val_gt_dir)
    print(run_training(config))


if __name__ == "__main__":
    main()
