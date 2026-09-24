# Mask2Former для сегментации строк текста

Репозиторий содержит обучение Mask2Former, экспорт обученной модели в ONNX и локальный Gradio-интерфейс. Подготовка исходных датасетов в репозиторий не входит.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Для обучения требуется CUDA. Gradio можно запускать как на CUDA, так и на CPU.

## Формат данных

Для каждого split нужны отдельные каталоги изображений и разметки. Файл разметки имеет то же имя, что и изображение, но расширение `.txt`. Каждая строка TXT описывает один полигон строки текста:

```text
x1,y1,x2,y2,x3,y3,x4,y4
```

Несколько наборов можно объединить, повторив соответствующие аргументы каталогов.

## Обучение

```powershell
.\.venv\Scripts\python.exe scripts\run_mask2former_page_line_training.py `
  --train-img-dir C:\data\train\images `
  --train-gt-dir C:\data\train\gt `
  --val-img-dir C:\data\val\images `
  --val-gt-dir C:\data\val\gt `
  --model-config C:\models\base_model `
  --output-dir C:\models\trained_model `
  --epochs 100
```

Скрипт автоматически продолжает обучение из совместимого checkpoint в `output-dir`. Для проверки конфигурации без обучения добавьте `--dry-run`.

## Gradio на ONNX Runtime

Интерфейс работает только через ONNX Runtime и не импортирует PyTorch или Transformers:

```powershell
.\.venv\Scripts\python.exe scripts\gradio_mask2former_lines.py `
  --model-dir .\mask2former_lines_swin_small_page_medium_ft100\best_f1
```

Для отдельного runtime-окружения достаточно установить `requirements-runtime.txt`. Обычный `onnxruntime` даёт CPU. Для переключения CPU/CUDA установите вместо него `onnxruntime-gpu`; этот пакет предоставляет оба execution provider. На CPU используется FP32-модель, на CUDA — FP16.

## Экспорт в ONNX

```powershell
.\.venv\Scripts\python.exe scripts\export_mask2former_onnx.py `
  --model-dir .\mask2former_lines_swin_small_page_medium_ft100\best_f1 `
  --output .\mask2former_lines_swin_small_page_medium_ft100\best_f1\model.onnx `
  --image-size 1024
```

ONNX-модель возвращает сырые `class_queries_logits` и `masks_queries_logits`. Восстановление масок исходного размера и фильтрация результатов выполняются отдельно.

Для отдельной FP16-версии добавьте `--fp16`. По умолчанию она сохраняется как `model.fp16.onnx` и предназначена прежде всего для `onnxruntime-gpu`.
