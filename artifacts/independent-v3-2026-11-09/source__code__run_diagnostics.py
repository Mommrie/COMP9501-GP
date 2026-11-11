"""Run bounded protocol-v2 learning checks with progress saved between jobs."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import read_config, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol-v2.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    cfg = read_config(args.config)
    cfg.update(train_samples=512, logging_steps=8)
    overfit = copy.deepcopy(cfg)
    overfit.update(train_samples=32, calibration_eval_samples=32, warmup_ratio=0., dropout=0.,
                   select_best_on_calibration=False)
    jobs = [("overfit_sst2", "sst2", overfit, ["--max-steps", "40"]),
            ("head_only_sst2", "sst2", cfg, ["--head-only"]),
            ("lora_sst2", "sst2", cfg, []), ("lora_rte", "rte", cfg, [])]
    status = {"state": "running", "seed": args.seed, "runs": {}}
    def publish():
        temporary = out / "status.tmp"
        write_json(temporary, status)
        os.replace(temporary, out / "status.json")
    publish()
    try:
        for name, task, job_cfg, extra in jobs:
            config = out / (name + "-config.json")
            write_json(config, job_cfg)
            status.update(current=name, started_at_epoch=time.time())
            publish()
            command = [sys.executable, str(root / "code/train_lora.py"), "--config", str(config),
                       "--task", task, "--seed", str(args.seed), "--output", str(out / name), *extra]
            start = time.perf_counter()
            with (out / (name + ".log")).open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{name} failed; see {name}.log")
            run = json.loads((out / name / "run.json").read_text())
            run["process_wall_seconds"] = time.perf_counter() - start
            status["runs"][name] = run
            if name == "overfit_sst2" and run["train_probe"]["train_probe_accuracy"] < .9:
                status.update(state="diagnostic_failed", reason="32-example training accuracy below 0.90; inspect before scaling")
                publish()
                return
            changes = run["parameter_change_from_initialization"]
            if changes["head_l2"] <= 0 or (not run["head_only"] and changes["lora_l2"] <= 0):
                raise RuntimeError(f"{name}: expected trainable parameters did not change")
            if name == "head_only_sst2":
                from safetensors.torch import load_file
                weights = load_file(str(out / name / "adapter/adapter_model.safetensors"))
                if any(t.count_nonzero() for key, t in weights.items() if key.endswith(".lora_B.weight")):
                    raise RuntimeError("Head-only control changed LoRA")
            publish()
        status.update(state="passed", current=None)
        publish()
    except Exception as error:
        status.update(state="error", error=str(error), traceback=traceback.format_exc())
        publish()
        raise


if __name__ == "__main__":
    main()
