import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from .data import build_dataset, collate_fn, labels_to_device
from .evaluation import evaluate_detection_metrics, evaluate_loss, write_epoch_previews
from .modeling import (
    build_model_from_scratch,
    build_processor,
    load_model_and_processor_for_resume,
    resolve_checkpoint_model_dir,
)


@dataclass
class TrainConfig:
    model_config: str = "facebook/mask2former-swin-tiny-coco-instance"
    resume_from_checkpoint: Optional[str] = None
    auto_resume: bool = False
    train_img_dir: str = "handwritten_essay_v2_east_lines/train_img"
    train_gt_dir: str = "handwritten_essay_v2_east_lines/train_gt"
    val_img_dir: str = "handwritten_essay_v2_east_lines/test_img"
    val_gt_dir: str = "handwritten_essay_v2_east_lines/test_gt"
    output_dir: str = "mask2former_lines_scratch"
    image_size: int = 512
    load_max_side: int = 1024
    epochs: int = 50
    batch_size: int = 1
    accumulation_steps: int = 8
    max_instances_per_image: int = 32
    lr: float = 1e-5
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    max_train_steps: Optional[int] = None
    max_val_steps: Optional[int] = None
    checkpoint_interval_steps: int = 0
    preview_count: int = 3
    preview_threshold: float = 0.25
    preview_mask_threshold: float = 0.5
    metric_images: int = 20
    metric_iou_threshold: float = 0.5
    gradient_checkpointing: bool = False
    augment: bool = True
    augmentation_strength: str = "medium"
    require_cuda: bool = False
    no_amp: bool = False
    cpu: bool = False


@dataclass
class TrainingResult:
    output_dir: str
    latest_checkpoint: str
    best_checkpoint: str
    start_epoch: int
    completed_epoch: int
    best_val_loss: float
    resumed_from: Optional[str]


def validate_config(config):
    if config.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    if config.image_size < 1:
        raise ValueError("image_size must be >= 1")
    if config.augmentation_strength not in {"light", "medium", "strong"}:
        raise ValueError("augmentation_strength must be one of: light, medium, strong")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not state:
        return
    if state.get("python") is not None:
        try:
            random.setstate(state["python"])
        except Exception as exc:
            print(f"python RNG state restore skipped: {exc}")
    if state.get("numpy") is not None:
        try:
            np.random.set_state(state["numpy"])
        except Exception as exc:
            print(f"numpy RNG state restore skipped: {exc}")
    if state.get("torch") is not None:
        torch_state = rng_state_to_cpu_byte_tensor(state["torch"])
        if torch_state is not None:
            try:
                torch.set_rng_state(torch_state)
            except Exception as exc:
                print(f"torch RNG state restore skipped: {exc}")
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = state["cuda"]
        if torch.is_tensor(cuda_states):
            cuda_states = [cuda_states]
        restored = [rng_state_to_cpu_byte_tensor(item) for item in cuda_states]
        restored = [item for item in restored if item is not None]
        if restored:
            try:
                torch.cuda.set_rng_state_all(restored)
            except Exception as exc:
                print(f"cuda RNG state restore skipped: {exc}")


def rng_state_to_cpu_byte_tensor(value):
    try:
        if torch.is_tensor(value):
            return value.detach().cpu().to(torch.uint8)
        if isinstance(value, (bytes, bytearray)):
            return torch.tensor(list(value), dtype=torch.uint8)
        if isinstance(value, (list, tuple)):
            return torch.tensor(value, dtype=torch.uint8)
    except Exception as exc:
        print(f"RNG tensor conversion skipped: {exc}")
    return None


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_training_state_metadata(checkpoint_dir):
    state_path = Path(checkpoint_dir) / "training_state.pt"
    if not state_path.exists():
        return None
    state = torch_load(state_path, map_location="cpu")
    return {
        "epoch": int(state.get("epoch", 1)),
        "step": int(state.get("step", 0)),
        "loss": float(state.get("loss", math.inf)),
    }


def find_resume_checkpoint(output_dir):
    output_dir = Path(output_dir)
    candidates = [output_dir / "latest", output_dir / "step_checkpoint", output_dir / "best"]
    available = []
    for candidate in candidates:
        metadata = read_training_state_metadata(candidate)
        if metadata is not None:
            try:
                resolve_checkpoint_model_dir(candidate)
            except FileNotFoundError:
                continue
            available.append((metadata["epoch"], metadata["step"], candidate))
    if not available:
        return None
    available.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return available[0][2]


def load_best_val_loss(output_root):
    metrics_path = Path(output_root) / "best" / "metrics.json"
    if not metrics_path.exists():
        return math.inf
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(metrics.get("val_loss", math.inf))
    except (OSError, ValueError, TypeError):
        return math.inf


def save_checkpoint(
    model,
    processor,
    optimizer,
    scaler,
    output_dir,
    epoch,
    step,
    loss,
    best_val_loss,
    train_loss=None,
    val_loss=None,
    completed_epoch=None,
    nested_model_dir=False,
):
    output_dir = Path(output_dir)
    temp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    model_dir = temp_dir / "model" if nested_model_dir else temp_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=False)
    processor.save_pretrained(model_dir)

    state = {
        "epoch": int(epoch),
        "step": int(step),
        "loss": float(loss),
        "best_val_loss": float(best_val_loss),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": capture_rng_state(),
    }
    torch.save(state, temp_dir / "training_state.pt")

    if train_loss is not None or val_loss is not None or completed_epoch is not None:
        metrics = {
            "epoch": completed_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "next_epoch": epoch,
            "resume_step": step,
        }
        (temp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    temp_dir.replace(output_dir)


def load_training_state(path, optimizer, scaler, device):
    state_path = Path(path) / "training_state.pt"
    if not state_path.exists():
        return 1, 0, math.inf
    state = torch_load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    if state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    restore_rng_state(state.get("rng_state"))
    return (
        int(state.get("epoch", 1)),
        int(state.get("step", 0)),
        float(state.get("best_val_loss", math.inf)),
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, processor, config, shuffle, epoch=None):
    kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "collate_fn": collate_fn(
            processor,
            config.image_size,
            config.max_instances_per_image,
            augment=bool(shuffle and config.augment),
            augmentation_strength=config.augmentation_strength,
        ),
        "worker_init_fn": seed_worker,
    }
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(config.seed + int(epoch or 0))
        kwargs["generator"] = generator
    return DataLoader(dataset, shuffle=shuffle, **kwargs)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    accumulation_steps,
    use_amp,
    processor=None,
    step_checkpoint_dir=None,
    checkpoint_interval_steps=0,
    skip_steps=0,
    max_steps=None,
    best_val_loss=math.inf,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    steps = 0
    last_checkpoint_step = int(skip_steps)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=True, dynamic_ncols=True)

    for step, batch in enumerate(progress, start=1):
        if step <= skip_steps:
            progress.set_postfix(skip=f"{step}/{skip_steps}")
            continue

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
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()
        did_optimizer_step = False
        if step % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            did_optimizer_step = True

        loss_value = float(loss.detach().cpu()) * accumulation_steps
        total_loss += loss_value
        steps += 1
        avg_loss = total_loss / steps
        progress.set_postfix(loss=f"{loss_value:.4f}", avg=f"{avg_loss:.4f}")

        should_save_step = (
            checkpoint_interval_steps
            and processor is not None
            and step_checkpoint_dir is not None
            and did_optimizer_step
            and step - last_checkpoint_step >= checkpoint_interval_steps
        )
        if should_save_step:
            save_checkpoint(
                model,
                processor,
                optimizer,
                scaler,
                step_checkpoint_dir,
                epoch=epoch,
                step=step,
                loss=avg_loss,
                best_val_loss=best_val_loss,
                nested_model_dir=True,
            )
            last_checkpoint_step = step
            print(f"step checkpoint saved: {step_checkpoint_dir}")

        if max_steps and steps >= max_steps:
            break

    if steps and steps % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, steps)


def run_training(config=None):
    if config is None:
        config = TrainConfig()
    elif isinstance(config, dict):
        config = TrainConfig(**config)
    validate_config(config)

    set_seed(config.seed)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = Path(config.resume_from_checkpoint) if config.resume_from_checkpoint else None
    if resume_checkpoint is None and config.auto_resume:
        resume_checkpoint = find_resume_checkpoint(output_root)

    cuda_available = torch.cuda.is_available()
    if config.require_cuda and (config.cpu or not cuda_available):
        raise RuntimeError(
            "CUDA is required by config.require_cuda=True, but this Python environment cannot use it. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
            f"cuda_available={cuda_available}, cpu={config.cpu}. "
            "Install a CUDA-enabled PyTorch build or set require_cuda=False."
        )

    device = torch.device("cuda" if cuda_available and not config.cpu else "cpu")
    use_amp = device.type == "cuda" and not config.no_amp
    print(f"device: {device}, amp: {use_amp}")

    if resume_checkpoint:
        print(f"loading local checkpoint: {resume_checkpoint}")
        processor, model = load_model_and_processor_for_resume(resume_checkpoint)
    else:
        print(f"initializing model from scratch from config: {config.model_config}")
        print("pretrained model weights are not loaded")
        processor = build_processor(config.model_config)
        model = build_model_from_scratch(
            config.model_config,
            num_labels=1,
            id2label={0: "text_line"},
            label2id={"text_line": 0},
        )

    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except ValueError as exc:
            print(f"gradient checkpointing skipped: {exc}")
    model.to(device)

    train_dataset = build_dataset(config.train_img_dir, config.train_gt_dir, config.load_max_side)
    val_dataset = build_dataset(config.val_img_dir, config.val_gt_dir, config.load_max_side)
    preview_dataset = val_dataset.datasets[0] if isinstance(val_dataset, ConcatDataset) else val_dataset
    val_loader = make_loader(val_dataset, processor, config, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = load_best_val_loss(output_root)
    start_epoch = 1
    resume_skip_steps = 0
    if resume_checkpoint:
        start_epoch, resume_skip_steps, state_best_val_loss = load_training_state(
            resume_checkpoint,
            optimizer,
            scaler,
            device,
        )
        best_val_loss = min(best_val_loss, state_best_val_loss)
        print(f"resuming training state: epoch={start_epoch}, skip_steps={resume_skip_steps}")

    print("training config:")
    print(json.dumps(asdict(config), indent=2, default=str))

    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, config.epochs + 1):
        start_time = time.time()
        skip_steps = resume_skip_steps if epoch == start_epoch else 0
        train_loader = make_loader(train_dataset, processor, config, shuffle=True, epoch=epoch)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            config.accumulation_steps,
            use_amp,
            processor=processor,
            step_checkpoint_dir=output_root / "step_checkpoint",
            checkpoint_interval_steps=config.checkpoint_interval_steps,
            skip_steps=skip_steps,
            max_steps=config.max_train_steps,
            best_val_loss=best_val_loss,
        )
        resume_skip_steps = 0
        val_loss = evaluate_loss(model, val_loader, device, use_amp, config.max_val_steps)
        completed_epoch = epoch
        elapsed = time.time() - start_time
        print(f"epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, time={elapsed / 60:.1f} min")

        next_epoch = epoch + 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                processor,
                optimizer,
                scaler,
                output_root / "best",
                epoch=next_epoch,
                step=0,
                loss=val_loss,
                best_val_loss=best_val_loss,
                train_loss=train_loss,
                val_loss=val_loss,
                completed_epoch=epoch,
            )
            print(f"saved best checkpoint: {output_root / 'best'}")

        save_checkpoint(
            model,
            processor,
            optimizer,
            scaler,
            output_root / "latest",
            epoch=next_epoch,
            step=0,
            loss=val_loss,
            best_val_loss=best_val_loss,
            train_loss=train_loss,
            val_loss=val_loss,
            completed_epoch=epoch,
        )

        metrics = evaluate_detection_metrics(
            model,
            processor,
            preview_dataset,
            device,
            config.image_size,
            config.preview_threshold,
            config.preview_mask_threshold,
            config.metric_iou_threshold,
            config.metric_images,
        )
        if metrics.get("skipped"):
            print("metrics skipped: metric_images=0")
        else:
            print(
                f"metrics@IoU{config.metric_iou_threshold:.2f}: "
                f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
                f"f1={metrics['f1']:.4f}, tp={metrics['tp']}, pred={metrics['pred']}, gt={metrics['gt']}"
            )

        write_epoch_previews(
            model,
            processor,
            preview_dataset,
            output_root / "epoch_previews",
            device,
            epoch,
            config.image_size,
            config.preview_threshold,
            config.preview_mask_threshold,
            config.preview_count,
        )

    return TrainingResult(
        output_dir=str(output_root),
        latest_checkpoint=str(output_root / "latest"),
        best_checkpoint=str(output_root / "best"),
        start_epoch=start_epoch,
        completed_epoch=completed_epoch,
        best_val_loss=best_val_loss,
        resumed_from=str(resume_checkpoint) if resume_checkpoint else None,
    )
