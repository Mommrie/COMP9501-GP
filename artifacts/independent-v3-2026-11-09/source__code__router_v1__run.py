import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'merge_study'))
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.linear_model import LogisticRegression
from common import read_config,write_json,tokenizer_for,dataset_for,dataset_manifest,batches,metrics
from engine import DeltaEvaluator,task_score


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    cfg,plan=read_config(args.config),read_config(args.plan);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(4);root=CODE.parent;tasks=cfg['tasks'];tokenizer=tokenizer_for(cfg)
    state={'state':'running','config':cfg,'plan':plan,'validation':{},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};start=time.perf_counter()
    def publish(current):
        state.update(current=current,wall_seconds=time.perf_counter()-start);write_json(out/'status.tmp',state);os.replace(out/'status.tmp',out/'status.json')
    try:
        engine=DeltaEvaluator.from_bank(cfg,root/'outputs/bank-v2-a100-s42');fit={};selection={};ids={}
        for task in tasks:
            data=dataset_for(task,cfg,tokenizer,'calibration',plan['total_calibration_samples']);n=plan['fit_samples_per_task']
            fit[task]=data.select(range(n));selection[task]=data.select(range(n,len(data)))
            manifest=dataset_manifest(task,cfg,'calibration',plan['total_calibration_samples']);ids[task]={'fit':manifest['sample_ids'][:n],'selection':manifest['sample_ids'][n:]}
        write_json(out/'split-ids.json',ids)
        @torch.inference_mode()
        def features(data):
            arrays=[];engine.model.eval()
            for batch in batches(data,tokenizer,cfg['batch_size']):
                batch.pop('labels');inputs={k:v.to(engine.device) for k,v in batch.items()}
                hidden=engine.model.model(**inputs).last_hidden_state.float();mask=inputs['attention_mask'].unsqueeze(-1)
                arrays.append(((hidden*mask).sum(1)/mask.sum(1).clamp_min(1)).cpu().numpy())
            return np.concatenate(arrays)
        publish('base-features-fit');xf=np.concatenate([features(fit[t]) for t in tasks]);yf=np.concatenate([np.full(len(fit[t]),i) for i,t in enumerate(tasks)])
        publish('base-features-selection');xs=np.concatenate([features(selection[t]) for t in tasks]);ys=np.concatenate([np.full(len(selection[t]),i) for i,t in enumerate(tasks)])
        scaler=StandardScaler().fit(xf);af=scaler.transform(xf);ass=scaler.transform(xs)
        prototypes=normalize(np.stack([af[yf==i].mean(0) for i in range(4)]))
        candidates=[];classifiers=[]
        for C in plan['linear_C']:
            model=LogisticRegression(C=C,max_iter=800,random_state=plan['router_seed']).fit(af,yf)
            candidates.append({'C':C,'source_selection_accuracy':float((model.predict(ass)==ys).mean())});classifiers.append(model)
        best=int(np.argmax([x['source_selection_accuracy'] for x in candidates]));linear=classifiers[best]
        locked={'linear_candidates':candidates,'linear_selected':candidates[best],
            'prototype_source_selection_accuracy':float(((normalize(ass)@prototypes.T).argmax(1)==ys).mean()),
            'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
            'linear_coef':linear.coef_.tolist(),'linear_intercept':linear.intercept_.tolist(),'prototypes':prototypes.tolist(),
            'locked_before_validation':True,'plan_sha256':hashlib.sha256(Path(args.plan).read_bytes()).hexdigest()}
        state['selection']=locked;write_json(out/'selection.json',locked)
        publish('base-features-validation-after-lock');data={t:dataset_for(t,cfg,tokenizer,'validation',0) for t in tasks}
        ftime=time.perf_counter();xv=np.concatenate([features(data[t]) for t in tasks]);state['validation_base_feature_seconds']=time.perf_counter()-ftime
        yv=np.concatenate([np.full(len(data[t]),i) for i,t in enumerate(tasks)]);av=scaler.transform(xv);rng=np.random.default_rng(plan['router_seed'])
        route={'prototype':(normalize(av)@prototypes.T).argmax(1),'linear':linear.predict(av),'task_id':yv.copy(),'random':rng.integers(0,4,len(yv))}
        formats=[]
        for t in tasks:
            for row in data[t]:
                prefix=tokenizer.decode(row['input_ids'][:10]);choices=[1,2] if prefix.startswith('Sentence A:') else [0,3]
                formats.append(int(rng.choice(choices)))
        route['format']=np.asarray(formats)
        np.savez_compressed(out/'features-and-routes.npz',fit=xf,selection=xs,validation=xv,source_task=yv,**{'route_'+k:v for k,v in route.items()})
        state['routing_accuracy']={k:float((v==yv).mean()) for k,v in route.items()}
        for seed in [42,43,44]:
            if seed!=42:
                del engine;gc.collect();torch.cuda.empty_cache()
                engine=DeltaEvaluator.from_bank(cfg,root/f'outputs/replication-s43-s44/bank-s{seed}')
            oldbank=root/('outputs/bank-v2-a100-s42' if seed==42 else f'outputs/replication-s43-s44/bank-s{seed}')
            old=read_config(oldbank/'status.json');state['validation'][str(seed)]={}
            for method in plan['methods']:
                values={};offset=0
                for t in tasks:
                    publish(f'validation:s{seed}:{method}:{t}');assign=route[method][offset:offset+len(data[t])];logits=np.zeros((len(data[t]),2));labels=np.asarray(data[t]['labels'])
                    for i,expert in enumerate(tasks):
                        selected=np.flatnonzero(assign==i)
                        if len(selected):
                            result=engine.evaluate(t,engine.banks[expert],data[t].select(selected.tolist()),tokenizer,cfg['batch_size']);logits[selected]=result['logits']
                    score=metrics((logits,labels));values[t]={'metrics':score,'primary_score':task_score(t,score),'samples':len(labels)}
                    if method=='task_id':assert score==old['evaluation'][t]['individual']['metrics']
                    offset+=len(data[t])
                state['validation'][str(seed)][method]={'tasks':values,'score':sum(v['primary_score'] for v in values.values())/4}
        state['state']='passed';publish(None);write_json(out/'summary.json',state)
    except Exception as e:
        state.update(state='error',error=str(e),traceback=traceback.format_exc());publish(state.get('current'));raise


if __name__=='__main__':main()
