"""Shared experiment protocol; calibration never uses validation examples."""
import importlib.metadata
import hashlib
import json
import platform
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

FIELDS = {"sst2": ("sentence",), "cola": ("sentence",),
          "mrpc": ("sentence1", "sentence2"), "rte": ("sentence1", "sentence2")}


def read_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment():
    packages = {}
    for name in ("torch", "transformers", "peft", "datasets", "accelerate", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result.update(gpu=props.name, vram_gib=props.total_memory / 2**30,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    return result


def tokenizer_for(cfg):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("model_revision"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def model_for(cfg):
    from transformers import AutoModelForSequenceClassification
    seed_all(cfg["base_seed"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=2, torch_dtype=torch.float32, revision=cfg.get("model_revision"))
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    return model


def selected_raw(task, cfg, split="train", limit=None):
    from datasets import load_dataset
    if task not in FIELDS:
        raise ValueError(f"Unsupported task: {task}")
    if split not in {"train", "calibration", "validation"}:
        raise ValueError(f"Unsupported split: {split}")
    raw = load_dataset("nyu-mll/glue", task, split="validation" if split == "validation" else "train",
                       revision=cfg.get("dataset_revision"))
    if split != "validation":
        parts = raw.train_test_split(test_size=cfg["calibration_fraction"], seed=cfg["base_seed"],
                                     stratify_by_column="label")
        raw = parts["test" if split == "calibration" else "train"]
    raw = raw.shuffle(seed=cfg["base_seed"])
    if limit:
        raw = raw.select(range(min(limit, len(raw))))
    return raw


def encode_example(task, cfg, tokenizer, example):
    """V2 preserves explicit sentence roles and truncates content longest-first.

    A missing input_format intentionally preserves archived protocol-v1 behavior.
    Headers contain no task name or ground-truth label.
    """
    mode = cfg.get("input_format", "legacy")
    texts = [example[field] for field in FIELDS[task]]
    if mode == "legacy":
        return tokenizer(*texts, truncation=True, max_length=cfg["max_length"])
    if mode != "sentence_fields_v2":
        raise ValueError(f"Unknown input_format: {mode}")
    if len(texts) == 1:
        return tokenizer("Text: " + texts[0], truncation=True, max_length=cfg["max_length"])
    prefix = tokenizer.encode("Sentence A: ", add_special_tokens=False)
    separator = tokenizer.encode("\nSentence B: ", add_special_tokens=False)
    left, right = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    budget = cfg["max_length"] - len(prefix) - len(separator) - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 2:
        raise ValueError("max_length is too small to retain both sentence fields")
    while len(left) + len(right) > budget:
        (left if len(left) > len(right) else right).pop()
    return tokenizer.prepare_for_model(prefix + left + separator + right,
        add_special_tokens=True, return_attention_mask=True, truncation=False)


def dataset_for(task, cfg, tokenizer, split="train", limit=None):
    raw = selected_raw(task, cfg, split, limit)
    def tokenize(batch):
        rows = [encode_example(task, cfg, tokenizer,
                {field: batch[field][i] for field in FIELDS[task]}) for i in range(len(batch["label"]))]
        result = {key: [row[key] for row in rows] for key in rows[0]}
        result["labels"] = batch["label"]
        return result
    return raw.map(tokenize, batched=True, remove_columns=raw.column_names)


def dataset_manifest(task, cfg, split="train", limit=None):
    raw = selected_raw(task, cfg, split, limit)
    ids = list(map(int, raw["idx"]))
    return {"dataset": "nyu-mll/glue", "task": task, "split": split,
            "input_format": cfg.get("input_format", "legacy"), "max_length": cfg["max_length"],
            "revision_requested": cfg.get("dataset_revision"), "fingerprint": raw._fingerprint,
            "sample_ids": ids, "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "label_names": raw.features["label"].names,
            "label_counts": dict(Counter(map(int, raw["label"])))}


def metrics(prediction):
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    logits, labels = prediction
    pred = np.argmax(logits, axis=-1)
    return {"accuracy": float(accuracy_score(labels, pred)),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, pred))}


def batches(dataset, tokenizer, batch_size):
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=DataCollatorWithPadding(tokenizer))
