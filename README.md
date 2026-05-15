# Mask2Former Text Line Training

Code for training, resuming, previewing, and exporting a Mask2Former text-line instance segmentation model.

## Install

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Layout

- `utils/data.py` - dataset parsing, polygon resizing, train augmentations, collate function.
- `utils/modeling.py` - model/processor creation and local checkpoint loading.
- `utils/training.py` - `TrainConfig`, `run_training`, checkpoint save/load, resume logic.
- `utils/evaluation.py` - validation loss, detection metrics, epoch previews.
- `utils/inference.py` - PyTorch inference helpers and mask drawing.
- `utils/postprocess.py` - mask NMS, IoU, bbox, contour helpers.
- `utils/onnx_runtime.py` - ONNX Runtime detector without `torch` or `transformers`.
- `scripts/train_mask2former_lines.py` - CLI training entrypoint.
- `scripts/train_school_iam_lines.py` - training entrypoint for `school_notebooks_RU_lines` + `coco_lines_iam_test`.
- `scripts/detect_mask2former_lines.py` - batch preview renderer.
- `scripts/export_mask2former_onnx.py` - ONNX export.
- `scripts/gradio_mask2former_lines.py` - Gradio demo.
- `notebooks/train_mask2former_lines.ipynb` - notebook start/resume training flow.

## Training From Scratch

Fresh training does not load pretrained model weights. The `--model-config` value is used only to load the architecture config and image processor. The model itself is created with `AutoModelForUniversalSegmentation.from_config(...)`, so weights are randomly initialized.

```powershell
.\.venv\Scripts\python.exe .\scripts\train_mask2former_lines.py `
  --model-config facebook/mask2former-swin-tiny-coco-instance `
  --train-img-dir "handwritten_essay_v2_east_lines/train_img;school_notebooks_RU_east_lines/train_img" `
  --train-gt-dir "handwritten_essay_v2_east_lines/train_gt;school_notebooks_RU_east_lines/train_gt" `
  --val-img-dir handwritten_essay_v2_east_lines/test_img `
  --val-gt-dir handwritten_essay_v2_east_lines/test_gt `
  --output-dir mask2former_lines_scratch `
  --augmentation-strength medium `
  --checkpoint-interval-steps 100
```

Train augmentations are enabled by default for the training loader only. Validation is not augmented. Use
`--augmentation-strength light|medium|strong` to control intensity, or `--no-augment` to disable it. The
augmentations keep image and instance mask in sync: small affine rotate/translate/scale/shear, brightness/contrast/color,
occasional grayscale, blur, noise, and JPEG artifacts.

For the prepared school + IAM outputs, use the dedicated launcher:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_school_iam_lines.py
```

By default this script trains on `school_notebooks_RU_lines/train_*` plus `coco_lines_iam_test/test_*`, and validates on
`school_notebooks_RU_lines/val_*`. It does not load pretrained model weights on a fresh run; it resumes only from local
checkpoints in `CONFIG["output_dir"]` when they exist. Edit the constants at the top of the script to change paths,
epochs, augmentation strength, or checkpoint settings.

Resume from the newest local checkpoint in `--output-dir`:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_mask2former_lines.py `
  --output-dir mask2former_lines_scratch `
  --auto-resume
```

Resume from an explicit local checkpoint:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_mask2former_lines.py `
  --resume-from-checkpoint mask2former_lines_scratch\latest `
  --output-dir mask2former_lines_scratch
```

Checkpoints:

- `output_dir/latest` - latest completed epoch, includes model files and `training_state.pt`.
- `output_dir/best` - best validation loss checkpoint, also includes `training_state.pt`.
- `output_dir/step_checkpoint` - optional mid-epoch checkpoint when `--checkpoint-interval-steps` is set.

## Notebook

Open `notebooks/train_mask2former_lines.ipynb`. It checks `mask2former_lines_scratch` for a local checkpoint and resumes if one exists. If no checkpoint exists, it starts from scratch with random model weights.

## Prepare COCO Boxes As Line Polygons

Use this when a COCO file contains word/fragment boxes and the training target should be one polygon per text line. Boxes are grouped into lines by vertical overlap / center proximity. By default each line polygon is the convex hull of all source polygon/box points in that line. A Shapely postprocess step removes overlaps so different line polygons do not intersect.

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_coco_lines.py `
  --coco-json "C:\Users\USER\Desktop\9мая\IAM\test_images_coco.json" `
  --image-dir "C:\Users\USER\Desktop\9мая\IAM\test_images" `
  --output-dir coco_lines_iam_test `
  --split-name test `
  --copy-mode hardlink `
  --hull-method convex `
  --preview-count 3
```

Useful options:

- `--hull-method convex` - default, robust convex hull behavior.
- `--hull-method concave` - optional, uses `concave_hull(points, concavity, length_threshold)`.
- `--hull-method alpha --alpha 1.5` - use `alphashape`.
- `--line-group-mode auto` - use `group_id` / `line_id` if present, otherwise group boxes/polygons geometrically.
- `--allow-intersections` - disables the non-overlap postprocess.
- `--non-overlap-gap 2.0` - subtracts a small extra gap between polygons.
- `--simplify-tolerance 1.0` - reduces polygon point count after clipping.
- `--max-hull-points 32` - limits dense source polygons before hull construction.
- `--exclude-file-names "bad_page.jpg;other.jpg"` - skip known problematic pages.
- `--start-index` / `--max-images` - process a split in chunks when a source JSON is large.

For `school_notebooks_RU`, convert train, val, and test splits:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_school_notebooks_lines.py `
  --dataset-dir "C:\shared\data0205\data02065\school_notebooks_RU" `
  --output-dir school_notebooks_RU_lines `
  --copy-mode hardlink `
  --line-group-mode auto `
  --include-category-names "pupil_text;pupil_comment" `
  --hull-method convex `
  --preview-count 3
```

The school notebooks converter reads `annotations_train.json`, `annotations_val.json`, and `annotations_test.json`. It searches images in both `train_images` and `test_images`, because these JSON splits can reference files from either folder. By default it keeps only student categories: `pupil_text` and `pupil_comment`.

The generic COCO converter reads either `bbox` or polygon `segmentation`. If annotations contain `group_id`, `groupId`, `group`, `line_id`, or `lineId`, those IDs are used as line groups.

The output can be used directly as validation data:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_mask2former_lines.py `
  --train-img-dir path\to\train_img `
  --train-gt-dir path\to\train_gt `
  --val-img-dir coco_lines_iam_test\test_img `
  --val-gt-dir coco_lines_iam_test\test_gt `
  --output-dir mask2former_lines_scratch
```

## ONNX Export And Runtime

```powershell
.\.venv\Scripts\python.exe .\scripts\export_mask2former_onnx.py `
  --model-dir mask2former_lines_scratch\best `
  --output mask2former_lines_scratch\best\model_512_dynamo.onnx `
  --image-size 512 `
  --opset 20
```

Use the ONNX-only detector from Python:

```python
import numpy as np
from PIL import Image
from utils.onnx_runtime import Mask2FormerOnnxLineDetector

detector = Mask2FormerOnnxLineDetector("mask2former_lines_scratch/best/model_512_dynamo.onnx")
image = np.asarray(Image.open("sample.jpg").convert("RGB"))
detections = detector.predict(image, threshold=0.25)
```

## PyTorch Inference

Batch preview:

```powershell
.\.venv\Scripts\python.exe .\scripts\detect_mask2former_lines.py `
  --model-dir mask2former_lines_scratch\best `
  --img-dir handwritten_essay_v2_east_lines\test_img `
  --output-dir mask2former_lines_preview
```

Gradio:

```powershell
.\.venv\Scripts\python.exe .\scripts\gradio_mask2former_lines.py `
  --model-dir mask2former_lines_scratch\best `
  --host 127.0.0.1 `
  --port 7860
```
