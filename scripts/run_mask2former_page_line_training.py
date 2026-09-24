"""Configurable launcher for Mask2Former text-line instance segmentation."""

import argparse
import os
import sys
import atexit
import threading
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def relaunch_in_venv():
    if os.environ.get("MASK2FORMER_SKIP_VENV_REEXEC") or not LOCAL_PYTHON.exists():
        return
    if Path(sys.executable).resolve() == LOCAL_PYTHON.resolve():
        return
    os.environ["MASK2FORMER_SKIP_VENV_REEXEC"] = "1"
    print(f"relaunching with {LOCAL_PYTHON}", flush=True)
    os.execv(
        str(LOCAL_PYTHON),
        [str(LOCAL_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


relaunch_in_venv()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TeeStream:
    """Write console output to the terminal and an append-only UTF-8 log."""

    def __init__(self, terminal, log_file, lock):
        self.terminal = terminal
        self.log_file = log_file
        self.lock = lock

    def write(self, value):
        with self.lock:
            self.terminal.write(value)
            self.terminal.flush()
            self.log_file.write(value)
            self.log_file.flush()
        return len(value)

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")


def enable_console_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_console.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    lock = threading.RLock()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file, lock)
    sys.stderr = TeeStream(original_stderr, log_file, lock)

    def close_log():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.flush()
        log_file.close()

    atexit.register(close_log)
    print("\n" + "=" * 80)
    print(f"training session started: {datetime.now().astimezone().isoformat()}")
    print(f"command: {' '.join(sys.argv)}")
    print(f"console log: {log_path}")
    print("=" * 80, flush=True)
    return log_path

import torch

from utils.training import TrainConfig, run_training


DEFAULT_OUTPUT_DIR = ROOT / "mask2former_lines_swin_small_page_medium_ft100"
DEFAULT_MODEL_CONFIG = ROOT / "pretrained" / "g_0_mask2former_lines_swin_small_page"

BASE_CONFIG = {
    # Fine-tune our generation-0 checkpoint. It has model weights but intentionally
    # no optimizer state, so this run starts a fresh optimizer from those weights.
    "pretrained": True,
    "auto_resume": True,
    "image_size": 1024,
    "load_max_side": 0,
    # This is the final epoch number, not the number of additional epochs.
    # With auto_resume=True, rerunning after epoch 100 continues at epoch 101.
    "epochs": 100,
    "batch_size": 1,
    "accumulation_steps": 8,
    "max_instances_per_image": 128,
    "num_queries": 150,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "seed": 42,
    "checkpoint_interval_steps": 250,
    "evaluation_interval_epochs": 5,
    "preview_count": 6,
    "preview_threshold": 0.15,
    "preview_mask_threshold": 0.40,
    "metric_images": 160,
    "metric_iou_threshold": 0.5,
    "gradient_checkpointing": False,
    "augment": True,
    "augmentation_strength": "medium",
    "require_cuda": True,
    "no_amp": False,
    "cpu": False,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-img-dir", action="append", required=True)
    parser.add_argument("--train-gt-dir", action="append", required=True)
    parser.add_argument("--val-img-dir", action="append", required=True)
    parser.add_argument("--val-gt-dir", action="append", required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=BASE_CONFIG["epochs"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def join_dirs(paths):
    return ";".join(str(Path(path).expanduser().resolve()) for path in paths)


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    enable_console_logging(output_dir)
    if BASE_CONFIG["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required; torch={torch.__version__}, cuda={torch.version.cuda}")
    print(f"python: {sys.executable}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    config_values = dict(BASE_CONFIG)
    config_values.update(
        model_config=str(args.model_config.expanduser().resolve()),
        output_dir=str(output_dir),
        epochs=args.epochs,
    )
    config = TrainConfig(
        **config_values,
        train_img_dir=join_dirs(args.train_img_dir),
        train_gt_dir=join_dirs(args.train_gt_dir),
        val_img_dir=join_dirs(args.val_img_dir),
        val_gt_dir=join_dirs(args.val_gt_dir),
    )
    print(f"epoch previews: {output_dir / 'epoch_previews'}")
    if args.dry_run or os.environ.get("MASK2FORMER_DRY_RUN"):
        print("dry run: training was not started")
        return
    print(run_training(config))


if __name__ == "__main__":
    main()
