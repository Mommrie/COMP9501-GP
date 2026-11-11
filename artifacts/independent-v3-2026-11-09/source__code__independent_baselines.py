"""Frozen weight-space baselines under the same coefficient-selection budget."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE/'merge_study'))
from common import dataset_for, environment, read_config, tokenizer_for, write_json
from engine import DeltaEvaluator, compact


def main():
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Run on Colab GPU')
    p = argparse.ArgumentParser()
    p.add_argument('--study', required=True)
    args = p.parse_args()
    study = Path(args.study)
    out = study/'weight-baselines'
    out.mkdir(exist_ok=False)
    cfg = read_config(study/'locked-config.json')
    plan_path = CODE.parent/'configs/merge-study-replication-v2.json'
    plan = read_config(plan_path)
    status = {'state': 'running', 'seeds': {}, 'plan': plan,
              'selection_role': 'coefficient_selection', 'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'note': 'Frozen prior 17-candidate grid; no new search after inspecting this round final scores.'}
    tokenizer = tokenizer_for(cfg)
    selection = {t:dataset_for(t,cfg,tokenizer,'coefficient_selection',0) for t in cfg['tasks']}
    def publish(current):
        status.update(current=current)
        write_json(out/'status.tmp',status)
        os.replace(out/'status.tmp',out/'status.json')
    def evaluate(engine, delta, data):
        values = {t:compact(engine.evaluate(t,delta,data[t],tokenizer,cfg['batch_size'])) for t in cfg['tasks']}
        return {'tasks': values, 'score': sum(v['primary_score'] for v in values.values())/len(values)}
    for seed in cfg['seeds']:
        bank = study/f'bank-s{seed}'
        record = {'selection': {}, 'validation': {}}
        status['seeds'][str(seed)] = record
        engine = DeltaEvaluator.from_bank(cfg,bank)
        for spec in plan['candidates']:
            publish(f'select-s{seed}:{spec["name"]}')
            record['selection'][spec['name']] = evaluate(engine,engine.merge(spec,seed=plan['merge_seed']),selection)
        chosen = {}
        for method in ['average','task_arithmetic','ties','dare']:
            candidates = [s for s in plan['candidates'] if s['method']==method]
            chosen[method] = max(candidates,key=lambda s:record['selection'][s['name']]['score'])
        write_json(out/f'selection-s{seed}.json',{'selected':chosen,'locked_epoch':time.time(),'before_final_load':True})
        record['selected'] = chosen
        final = {t:dataset_for(t,cfg,tokenizer,'validation',0) for t in cfg['tasks']}
        for method,spec in chosen.items():
            publish(f'final-s{seed}:{method}')
            record['validation'][method] = evaluate(engine,engine.merge(spec,seed=plan['merge_seed']),final)
        del engine
        torch.cuda.empty_cache()
    status.update(state='passed',environment=environment())
    publish(None)
    write_json(out/'summary.json',status)


if __name__ == '__main__':
    main()
