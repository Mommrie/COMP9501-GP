"""Run isolated seed banks concurrently, then apply the frozen study protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import read_config, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--plan",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[43,44])
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(out)
    if len(set(args.seeds))!=len(args.seeds):raise ValueError("Duplicate seed")
    config,plan=Path(args.config).resolve(),Path(args.plan).resolve()
    out.mkdir(parents=True)
    state={"state":"running","seeds":args.seeds,"runs":{},"workers":args.workers,
           "config":read_config(config),"plan":read_config(plan)}
    lock=threading.Lock()
    def update(seed,**values):
        with lock:
            state["runs"].setdefault(str(seed),{}).update(values)
            write_json(out/"status.tmp",state)
            os.replace(out/"status.tmp",out/"status.json")
    def job(seed):
        bank=out/f"bank-s{seed}"
        study=out/f"study-s{seed}"
        # Bound host threads when two processes share one GPU/Colab VM.
        env={**os.environ,"OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}
        start=time.perf_counter()
        for stage,command in [
            ("bank",[sys.executable,"code/run_bank.py","--config",str(config),"--output",str(bank),"--seed",str(seed)]),
            ("study",[sys.executable,"code/merge_study/run.py","--config",str(config),"--plan",str(plan),"--bank",str(bank),"--output",str(study)])]:
            update(seed,stage=stage,bank=str(bank),study=str(study))
            with (out/f"s{seed}-{stage}.log").open("w",encoding="utf-8") as log:
                result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,env=env)
            if result.returncode:
                update(seed,stage="error",error=f"{stage} return code {result.returncode}")
                raise RuntimeError(f"Seed {seed} {stage} failed")
        update(seed,stage="passed",wall_seconds=time.perf_counter()-start)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures=[executor.submit(job,seed) for seed in args.seeds]
            for future in as_completed(futures):future.result()
        state["state"]="passed"
    except Exception as error:
        state.update(state="error",error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        with lock:
            write_json(out/"status.tmp",state)
            os.replace(out/"status.tmp",out/"status.json")


if __name__=="__main__":main()
