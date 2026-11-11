"""Merge materialized LoRA deltas, never average the low-rank factors."""
import argparse
import json
import math
from pathlib import Path

import torch


def adapter_deltas(path):
    from safetensors.torch import load_file
    path = Path(path)
    cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("use_dora") or cfg.get("fan_in_fan_out") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError("Pilot supports ordinary LoRA with constant rank/alpha only")
    scale = cfg["lora_alpha"] / (math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
    state = load_file(str(path / "adapter_model.safetensors"))
    deltas = {}
    for name, a in state.items():
        if name.endswith(".lora_A.weight"):
            stem = name[:-len(".lora_A.weight")]
            key = stem.removeprefix("base_model.model.") + ".weight"
            deltas[key] = (state[stem + ".lora_B.weight"].float() @ a.float()) * scale
    if not deltas:
        raise ValueError(f"No LoRA tensors found in {path}")
    return deltas


def merge_tensors(tensors, method="average", scale=1.0, density=0.2, seed=42):
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    x = torch.stack([t.float() for t in tensors])
    if not torch.isfinite(x).all():
        raise ValueError("Non-finite delta")
    if method == "average":
        return x.mean(0) * scale
    if method == "task_arithmetic":
        return x.sum(0) * scale
    if method == "dare":
        generator = torch.Generator(device=x.device).manual_seed(seed)
        mask = torch.rand(x.shape, generator=generator, device=x.device) < density
        return (x * mask / density).sum(0) * scale
    if method != "ties":
        raise ValueError(method)
    flat = x.flatten(1)
    keep = max(1, math.ceil(density * flat.shape[1]))
    indices = flat.abs().topk(keep, dim=1).indices
    trimmed = torch.zeros_like(flat).scatter_(1, indices, flat.gather(1, indices))
    elected = trimmed.sum(0).sign()
    mask = (trimmed.sign() == elected) & (trimmed != 0)
    merged = (trimmed * mask).sum(0) / mask.sum(0).clamp_min(1)
    return merged.reshape(x.shape[1:]) * scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--method", choices=["average", "task_arithmetic", "ties", "dare"], default="average")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    banks = [adapter_deltas(path) for path in args.adapters]
    configs = [json.loads((Path(path) / "adapter_config.json").read_text()) for path in args.adapters]
    if len({c["base_model_name_or_path"] for c in configs}) != 1:
        raise ValueError("Adapters must use the same base model")
    if any(bank.keys() != banks[0].keys() for bank in banks):
        raise ValueError("Adapter target layers differ")
    merged = {key: merge_tensors([bank[key] for bank in banks], args.method, args.scale,
                                args.density, args.seed + i) for i, key in enumerate(sorted(banks[0]))}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"deltas": merged, "metadata": {**vars(args),
               "base_model": configs[0]["base_model_name_or_path"],
               "ties_variant": "per-matrix top-k, magnitude sign election, disjoint mean"}}, args.output)
    print(f"MERGE_COMPLETE: {args.output}")


if __name__ == "__main__":
    main()
