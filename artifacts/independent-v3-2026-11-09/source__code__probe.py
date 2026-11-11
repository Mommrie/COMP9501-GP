"""Compare last-layer mean-pooled delta activations on identical held-out train inputs."""
import argparse
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

from common import batches, dataset_for, model_for, read_config, tokenizer_for, write_json


def collect(model, loader, device):
    rows = []
    model = model.to(device).eval()
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None)
            inputs = {k: v.to(device) for k, v in batch.items()}
            hidden = model(**inputs, output_hidden_states=True).hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1)
            rows.append(((hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu())
    model.cpu()
    return torch.cat(rows)


def cosine_matrix(features):
    flat = torch.stack([f.flatten().float() for f in features])
    normalized = F.normalize(flat, dim=1)
    return normalized @ normalized.T


def main():
    from datasets import concatenate_datasets
    from peft import PeftModel
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(args.output)
    cfg = read_config(args.config)
    tokenizer = tokenizer_for(cfg)
    data = concatenate_datasets([dataset_for(task, cfg, tokenizer, "calibration",
        cfg["probe_samples_per_task"]) for task in cfg["tasks"]])
    loader = batches(data, tokenizer, cfg["batch_size"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_for(cfg)
    base = collect(model, loader, device)
    del model
    gc.collect()
    raw, deltas = [], []
    for adapter in args.adapters:
        model = PeftModel.from_pretrained(model_for(cfg), adapter)
        activation = collect(model, loader, device)
        raw.append(activation)
        deltas.append(activation - base)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out.mkdir(parents=True)
    torch.save({"base": base, "delta": torch.stack(deltas)}, out / "features.pt")
    write_json(out / "similarity.json", {"adapters": args.adapters, "samples": len(data),
        "raw_cosine": cosine_matrix(raw).tolist(), "delta_cosine": cosine_matrix(deltas).tolist(),
        "delta_norms": [float(d.norm()) for d in deltas],
        "protocol": "last hidden layer; mask-aware token mean; flatten common examples; train calibration only"})
    print(f"PROBE_COMPLETE: {out}")


if __name__ == "__main__":
    main()
