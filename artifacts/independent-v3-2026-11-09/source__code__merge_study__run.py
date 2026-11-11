"""Calibration-only baseline selection followed by locked validation diagnostics."""
import argparse
import gc
import hashlib
import itertools
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from common import batches, dataset_for, dataset_manifest, environment, read_config, tokenizer_for, write_json
from evaluate import load_evaluation_model
from probe import cosine_matrix
from engine import DeltaEvaluator, compact


def correlations(x, y):
    from scipy.stats import rankdata
    x, y = np.asarray(x), np.asarray(y)
    def pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None
    return {"pearson": pearson(x, y), "spearman": pearson(rankdata(x), rankdata(y))}


def weight_cosines(banks, tasks):
    norms = {t: sum(float(torch.dot(x.flatten(), x.flatten())) for x in banks[t].values()) ** .5
             for t in tasks}
    return [[sum(float(torch.dot(banks[a][k].flatten(), banks[b][k].flatten())) for k in banks[a]) /
             (norms[a] * norms[b]) for b in tasks] for a in tasks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg, plan = read_config(args.config), read_config(args.plan)
    bank, out = Path(args.bank).resolve(), Path(args.output).resolve()
    previous = read_config(bank / "status.json")
    if previous["state"] != "passed" or previous["config"] != cfg:
        raise ValueError("Bank protocol differs from study config")
    if plan["selection_split"] != "calibration":
        raise ValueError("Selection must use calibration")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    status = {"state": "running", "config": cfg, "plan": plan, "seed": previous["seed"],
              "calibration": {}, "validation": {}, "pairs": [],
              "source_sha256": {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path(__file__).parent.glob("*.py")}}
    start = time.perf_counter()

    def publish(current):
        status.update(current=current, wall_seconds=time.perf_counter() - start)
        write_json(out / "status.tmp", status)
        os.replace(out / "status.tmp", out / "status.json")

    def evaluate(task, delta, data, tag):
        publish(tag + ":" + task)
        return compact(engine.evaluate(task, delta, data, tokenizer, cfg["batch_size"]))

    try:
        publish("load-calibration")
        tokenizer = tokenizer_for(cfg)
        tasks = cfg["tasks"]
        calibration = {t: dataset_for(t, cfg, tokenizer, "calibration", plan["selection_samples_per_task"])
                       for t in tasks}
        manifests = {t: dataset_manifest(t, cfg, "calibration", plan["selection_samples_per_task"])
                     for t in tasks}
        for t in tasks:
            trained = read_config(bank / t / "data-manifest.json")
            if set(trained["train"]["sample_ids"]) & set(manifests[t]["sample_ids"]):
                raise ValueError("Calibration overlaps training")
        write_json(out / "calibration-manifests.json", manifests)
        publish("load-engine")
        engine = DeltaEvaluator.from_bank(cfg, bank)
        average_spec = plan["candidates"][0]
        average = engine.merge(average_spec, seed=plan["merge_seed"])
        status["parity"] = []
        # Direct logits, not just aggregate metrics, cross-check the new reusable engine.
        for task, merged in [(t, False) for t in tasks] + [(tasks[0], True)]:
            publish("parity:" + task + (":average" if merged else ":individual"))
            delta = average if merged else engine.banks[task]
            engine.set_state(task, delta)
            batch = next(iter(batches(calibration[task].select(range(8)), tokenizer, 8)))
            batch.pop("labels")
            inputs = {k:v.to(engine.device) for k,v in batch.items()}
            reference = load_evaluation_model(cfg, str(bank/task/"adapter"),
                                             str(bank/"average.pt") if merged else None).to(engine.device)
            with torch.inference_mode():
                expected = reference(**inputs).logits
                actual = engine.model(**inputs).logits
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            status["parity"].append({"task":task,"average":merged,
                                    "maximum_logit_error":float((actual-expected).abs().max())})
            del reference
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        del average
        # The whole declared grid is evaluated before any validation dataset is loaded.
        for spec in plan["candidates"]:
            publish("build:" + spec["name"])
            delta = engine.merge(spec, seed=plan["merge_seed"])
            values = {t:evaluate(t, delta, calibration[t], "calibration:"+spec["name"]) for t in tasks}
            status["calibration"][spec["name"]] = {"candidate":spec,"tasks":values,
                "macro_primary_score":float(np.mean([v["primary_score"] for v in values.values()]))}
            del delta
        if status["calibration"]["average"]["tasks"]:
            for t in tasks:
                assert status["calibration"]["average"]["tasks"][t]["metrics"] == status["calibration"]["ta025"]["tasks"][t]["metrics"]
        selected = {}
        for method in ["average","task_arithmetic","ties","dare"]:
            candidates = [s for s in plan["candidates"] if s["method"] == method]
            selected[method] = max(candidates, key=lambda s:status["calibration"][s["name"]]["macro_primary_score"])
        lock = {"selected":selected,"criterion":plan["selection_score"],"split":"calibration",
                "plan_sha256":hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
                "locked_before_validation_load":True,"locked_at_epoch":time.time()}
        write_json(out / "selection.json", lock)
        status["selection"] = lock
        publish("load-validation-after-lock")
        validation = {t:dataset_for(t, cfg, tokenizer, "validation", 0) for t in tasks}
        write_json(out / "validation-manifests.json", {t:dataset_manifest(t,cfg,"validation",0) for t in tasks})
        individual = {t:evaluate(t, engine.banks[t], validation[t], "validation:individual") for t in tasks}
        status["validation"]["individual"] = {"tasks":individual,
            "macro_primary_score":float(np.mean([v["primary_score"] for v in individual.values()]))}
        for t in tasks:
            assert individual[t]["metrics"] == previous["evaluation"][t]["individual"]["metrics"], "Individual parity changed"
        for method, spec in selected.items():
            publish("build-validation:"+method)
            delta = engine.merge(spec, seed=plan["merge_seed"])
            values = {t:evaluate(t,delta,validation[t],"validation:"+method) for t in tasks}
            status["validation"][method] = {"candidate":spec,"tasks":values,
                "macro_primary_score":float(np.mean([v["primary_score"] for v in values.values()]))}
            if method == "average":
                for t in tasks:
                    assert values[t]["metrics"] == previous["evaluation"][t]["average"]["metrics"], "Average parity changed"
            del delta
        probe = read_config(bank/"probe/similarity.json")
        weight = weight_cosines(engine.banks, tasks)
        features = torch.load(bank/"probe/features.pt",map_location="cpu",weights_only=True)
        count = cfg["probe_samples_per_task"]
        assert len(features["base"]) == count * len(tasks)
        pools = {t:cosine_matrix([x[i*count:(i+1)*count] for x in features["delta"]]).tolist()
                 for i,t in enumerate(tasks)}
        for i,j in itertools.combinations(range(len(tasks)),2):
            a,b = tasks[i],tasks[j]
            delta = engine.merge(average_spec,[a,b],plan["merge_seed"])
            values = {t:evaluate(t,delta,validation[t],f"pair:{a}+{b}") for t in [a,b]}
            accuracy_drop = {t:individual[t]["metrics"]["accuracy"]-values[t]["metrics"]["accuracy"] for t in [a,b]}
            primary_drop = {t:individual[t]["primary_score"]-values[t]["primary_score"] for t in [a,b]}
            status["pairs"].append({"tasks":[a,b],"metrics":values,"accuracy_drop":accuracy_drop,
                "mean_accuracy_drop":float(np.mean(list(accuracy_drop.values()))),
                "worst_accuracy_drop":max(accuracy_drop.values()),
                "mean_primary_drop":float(np.mean(list(primary_drop.values()))),
                "delta_cosine":probe["delta_cosine"][i][j],"raw_cosine":probe["raw_cosine"][i][j],
                "weight_cosine":weight[i][j],"delta_cosine_by_probe_source":{t:m[i][j] for t,m in pools.items()},
                "same_input_format_family":(a in {"mrpc","rte"}) == (b in {"mrpc","rte"})})
            del delta
        pairs = status["pairs"]
        y = [p["mean_accuracy_drop"] for p in pairs]
        status["correlations"] = {signal:correlations([p[signal] for p in pairs],y)
                                    for signal in ["delta_cosine","raw_cosine","weight_cosine"]}
        status["correlations"]["delta_vs_primary_drop"] = correlations(
            [p["delta_cosine"] for p in pairs],[p["mean_primary_drop"] for p in pairs])
        status["correlations"]["by_probe_source"] = {t:correlations(
            [p["delta_cosine_by_probe_source"][t] for p in pairs],y) for t in tasks}
        status["correlations"]["leave_one_pair_out"] = [{"omitted":p["tasks"],
            **correlations([v["delta_cosine"] for k,v in enumerate(pairs) if k!=i],
                           [v["mean_accuracy_drop"] for k,v in enumerate(pairs) if k!=i])}
            for i,p in enumerate(pairs)]
        status.update(state="passed",environment=environment())
        publish(None)
        write_json(out/"summary.json",status)
        print("MERGE_STUDY_COMPLETE",out,flush=True)
    except Exception as error:
        status.update(state="error",error=str(error),traceback=traceback.format_exc())
        publish(status.get("current"))
        raise


if __name__ == "__main__":
    main()
