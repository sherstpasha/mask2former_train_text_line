from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoImageProcessor, AutoModelForUniversalSegmentation


DEFAULT_ID2LABEL = {0: "text_line"}
DEFAULT_LABEL2ID = {"text_line": 0}


def resolve_checkpoint_model_dir(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")
    nested_model_dir = checkpoint_dir / "model"
    if nested_model_dir.exists():
        return nested_model_dir
    return checkpoint_dir


def build_processor(model_config):
    return AutoImageProcessor.from_pretrained(model_config)


def build_model_from_scratch(model_config, num_labels=1, id2label=None, label2id=None):
    config = AutoConfig.from_pretrained(model_config)
    config.num_labels = int(num_labels)
    config.id2label = id2label or DEFAULT_ID2LABEL
    config.label2id = label2id or DEFAULT_LABEL2ID
    return AutoModelForUniversalSegmentation.from_config(config)


def build_model_from_pretrained(
    model_config,
    num_labels=1,
    id2label=None,
    label2id=None,
    num_queries=None,
):
    config = AutoConfig.from_pretrained(model_config)
    config.num_labels = int(num_labels)
    config.id2label = id2label or DEFAULT_ID2LABEL
    config.label2id = label2id or DEFAULT_LABEL2ID
    model = AutoModelForUniversalSegmentation.from_pretrained(
        model_config,
        config=config,
        ignore_mismatched_sizes=True,
    )
    if num_queries is not None and int(num_queries) != int(model.config.num_queries):
        transformer = model.model.transformer_module
        target_queries = int(num_queries)
        for attribute in ("queries_features", "queries_embedder"):
            old_embedding = getattr(transformer, attribute)
            new_embedding = nn.Embedding(target_queries, old_embedding.embedding_dim)
            nn.init.normal_(new_embedding.weight, mean=0.0, std=float(getattr(config, "init_std", 0.02)))
            copied = min(len(old_embedding.weight), target_queries)
            with torch.no_grad():
                new_embedding.weight[:copied].copy_(old_embedding.weight[:copied])
            setattr(transformer, attribute, new_embedding)
        model.config.num_queries = target_queries
    return model


def load_model_and_processor_for_resume(checkpoint_dir):
    model_dir = resolve_checkpoint_model_dir(checkpoint_dir)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForUniversalSegmentation.from_pretrained(model_dir)
    return processor, model
