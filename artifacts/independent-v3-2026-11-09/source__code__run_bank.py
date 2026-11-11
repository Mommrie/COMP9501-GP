"""Build one seed's adapter bank, then evaluate individual/average and probe."""
import argparse
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = Path(args.config).resolve()
    cfg = read_config(config)
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    write_json(out / "config.json", cfg)
    status = {"state": "running", "seed": args.seed, "config": cfg,
              "training": {}, "evaluation": {}, "jobs": []}

    def publish():
        write_json(out / "status.tmp", status)
        os.replace(out / "status.tmp", out / "status.json")

    def run(name, script, *arguments):
        status.update(current=name, started_at_epoch=time.time())
        publish()
        start = time.perf_counter()
        with (out / (name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(root / "code" / script), *map(str, arguments)],
                                    cwd=root, stdout=log, stderr=subprocess.STDOUT)
        status["jobs"].append({"name": name, "returncode": result.returncode,
                               "wall_seconds": time.perf_counter() - start})
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {name}.log")

    publish()
    try:
        adapters = []
        for task in cfg["tasks"]:
            task_out = out / task
            run("train-" + task, "train_lora.py", "--config", config, "--task", task,
                "--seed", args.seed, "--output", task_out)
            status["training"][task] = read_config(task_out / "run.json")
            adapters.append(task_out / "adapter")
            publish()
        merge = out / "average.pt"
        run("merge-average", "merge.py", "--adapters", *adapters, "--method", "average",
            "--output", merge)
        for task, adapter in zip(cfg["tasks"], adapters):
            status["evaluation"][task] = {}
            for kind in ["individual", "average"]:
                result_path = out / f"{kind}-{task}.json"
                extra = ["--merge", merge] if kind == "average" else []
                run(f"eval-{kind}-{task}", "evaluate.py", "--config", config,
                    "--task", task, "--adapter", adapter, *extra, "--output", result_path)
                status["evaluation"][task][kind] = read_config(result_path)
                publish()
        run("probe", "probe.py", "--config", config, "--adapters", *adapters,
            "--output", out / "probe")
        status["probe"] = read_config(out / "probe/similarity.json")
        status.update(state="passed", current=None)
        publish()
    except Exception as error:
        status.update(state="error", error=str(error), traceback=traceback.format_exc())
        publish()
        raise


if __name__ == "__main__":
    main()
