"""Task-aware evaluation with each task's original classification head."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from common import batches, dataset_for, environment, metrics, model_for, read_config, tokenizer_for, write_json
from merge import adapter_deltas


def load_evaluation_model(cfg, adapter, merge=None):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model_for(cfg), adapter).merge_and_unload()
    if merge:
        payload = torch.load(merge, map_location="cpu", weights_only=True)
        if payload["metadata"]["base_model"] != cfg["model"]:
            raise ValueError("Merge/base model mismatch")
        own = adapter_deltas(adapter)
        merged = payload["deltas"]
        if own.keys() != merged.keys():
            raise ValueError("Merge target layers differ from evaluation adapter")
        params = dict(model.named_parameters())
        with torch.no_grad():
            for key in own:
                params[key].add_(merged[key].to(params[key]) - own[key].to(params[key]))
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--task", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--merge")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    cfg = read_config(args.config)
    tokenizer = tokenizer_for(cfg)
    dataset = dataset_for(args.task, cfg, tokenizer, "validation", cfg["eval_samples"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_evaluation_model(cfg, args.adapter, args.merge).to(device)
    logits, labels = [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in batches(dataset, tokenizer, cfg["batch_size"]):
            labels.append(batch.pop("labels").numpy())
            logits.append(model(**{k: v.to(device) for k, v in batch.items()}).logits.float().cpu().numpy())
    write_json(args.output, {"task": args.task, "adapter": args.adapter, "merge": args.merge,
        "metrics": metrics((np.concatenate(logits), np.concatenate(labels))),
        "samples": len(dataset), "seconds": time.perf_counter() - start, "environment": environment(),
        "protocol": "task ID selects the original task head; only backbone deltas merged"})
    print(f"EVAL_COMPLETE: {args.output}")


if __name__ == "__main__":
    main()
