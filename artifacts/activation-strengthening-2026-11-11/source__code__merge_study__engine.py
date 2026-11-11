"""Independent delta state replacement with immutable base snapshots."""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from common import batches, metrics, model_for
from merge import adapter_deltas, merge_tensors


def task_score(task, result):
    if task == "cola":
        return result["mcc"]
    if task == "mrpc":
        return (result["accuracy"] + result["f1"]) / 2
    return result["accuracy"]


class DeltaEvaluator:
    def __init__(self, model, banks, heads, device):
        self.model = model.to(device).eval()
        self.device = device
        self.banks = banks
        self.heads = heads
        self.params = dict(self.model.named_parameters())
        keys = next(iter(banks.values())).keys()
        if any(bank.keys() != keys for bank in banks.values()):
            raise ValueError("Adapter layer mismatch")
        self.base = {key: self.params[key].detach().clone() for key in keys}

    @classmethod
    def from_bank(cls, cfg, bank_path):
        from safetensors.torch import load_file
        banks, heads = {}, {}
        for task in cfg["tasks"]:
            adapter = Path(bank_path) / task / "adapter"
            banks[task] = adapter_deltas(adapter)
            state = load_file(str(adapter / "adapter_model.safetensors"))
            keys = [key for key in state if key.endswith("score.weight")]
            if len(keys) != 1:
                raise ValueError(f"Expected one classification head: {keys}")
            heads[task] = state[keys[0]].float()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(model_for(cfg), banks, heads, device)

    def merge(self, candidate, tasks=None, seed=42):
        tasks = tasks or list(self.banks)
        return {key: merge_tensors([self.banks[t][key] for t in tasks], candidate["method"],
                    candidate["scale"], candidate["density"], seed + index)
                for index, key in enumerate(sorted(self.base))}

    def set_state(self, task, deltas):
        if deltas.keys() != self.base.keys():
            raise ValueError("Merge target mismatch")
        with torch.no_grad():
            for key, base in self.base.items():
                self.params[key].copy_(base + deltas[key].to(base))
            self.params["score.weight"].copy_(self.heads[task].to(self.params["score.weight"]))

    def evaluate(self, task, deltas, data, tokenizer, batch_size):
        self.set_state(task, deltas)
        logits, labels = [], []
        start = time.perf_counter()
        with torch.inference_mode():
            for batch in batches(data, tokenizer, batch_size):
                labels.append(batch.pop("labels").numpy())
                inputs = {key: value.to(self.device) for key, value in batch.items()}
                logits.append(self.model(**inputs).logits.float().cpu().numpy())
        prediction = np.concatenate(logits)
        labels = np.concatenate(labels)
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite logits")
        result = metrics((prediction, labels))
        return {"metrics": result, "primary_score": task_score(task, result), "samples": len(data),
                "seconds": time.perf_counter() - start, "logits": prediction, "labels": labels}


def compact(result):
    return {key: value for key, value in result.items() if key not in {"logits", "labels"}}
