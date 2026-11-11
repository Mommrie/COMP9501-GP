"""Frozen source-router sensitivity to headers; not a new holdout evaluation."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch

from common import (read_config, write_json, tokenizer_for, model_for,
                    selected_raw, encode_example, batches, environment)


def strip_headers(task, ids, tokenizer):
    """Remove only constructed field labels, preserving retained content tokens."""
    if task in ('sst2', 'cola'):
        prefix = tokenizer.encode('Text:', add_special_tokens=False)
        assert ids[:len(prefix)] == prefix
        return ids[len(prefix):], len(prefix)
    prefix = tokenizer.encode('Sentence A: ', add_special_tokens=False)
    separator = tokenizer.encode('\nSentence B: ', add_special_tokens=False)
    assert ids[:len(prefix)] == prefix
    hits = [i for i in range(len(prefix),len(ids)-len(separator)+1)
            if ids[i:i+len(separator)] == separator]
    assert len(hits) == 1, 'Ambiguous constructed field separator'
    i = hits[0]
    newline = tokenizer.encode('\n', add_special_tokens=False)
    value = ids[len(prefix):i] + newline + ids[i+len(separator):]
    return value, len(prefix)+len(separator)-len(newline)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/protocol-v2-a100.json')
    p.add_argument('--samples-per-task',type=int,default=128)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Run in Colab GPU, not local CPU')
    root=Path(__file__).resolve().parent.parent
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=read_config(args.config)
    plan={'samples_per_task':args.samples_per_task,'tasks':cfg['tasks'],
          'sampling':'first N of protocol-v2 deterministic shuffled validation',
          'variants':['original','headerless','common_header'],
          'common_header':'Input: prepended to headerless token content',
          'fitting':'none; archived scaler and source classifier remain frozen',
          'scope':'source-task recognition only; classification heads and class labels not used',
          'limitation':'Previously explored validation; header edits shift positions/lengths and are distribution shifts, not proof of sole reliance on formatting.',
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(out/'plan.json',plan)
    torch.set_num_threads(4)
    tokenizer=tokenizer_for(cfg);model=model_for(cfg).cuda().eval()
    archived=read_config(root/'artifacts/router-v1-summary-2026-10-25.json')['selection']
    mean,scale,coef,intercept=[np.array(archived[k]) for k in
                             ('scaler_mean','scaler_scale','linear_coef','linear_intercept')]
    common=tokenizer.encode('Input:',add_special_tokens=False)
    predictions={v:[] for v in plan['variants']};truth=[];records=[];manifests={}
    @torch.inference_mode()
    def predict(rows):
        result=[]
        for batch in batches(rows,tokenizer,cfg['batch_size']):
            batch.pop('labels')
            inputs={k:v.cuda() for k,v in batch.items()}
            hidden=model.model(**inputs).last_hidden_state.float()
            mask=inputs['attention_mask'].unsqueeze(-1)
            x=((hidden*mask).sum(1)/mask.sum(1).clamp_min(1)).cpu().numpy()
            assert np.isfinite(x).all()
            result.extend(((((x-mean)/scale)@coef.T)+intercept).argmax(1).tolist())
        return result
    for source,task in enumerate(cfg['tasks']):
        raw=selected_raw(task,cfg,'validation',args.samples_per_task)
        ids=[int(x) for x in raw['idx']]
        manifests[task]={'ids':ids,'ids_sha256':hashlib.sha256(str(ids).encode()).hexdigest(),
                         'dataset_fingerprint':raw._fingerprint}
        rows={v:[] for v in plan['variants']}
        for example in raw:
            original=encode_example(task,cfg,tokenizer,example)['input_ids']
            neutral,removed=strip_headers(task,original,tokenizer)
            assert removed>0 and len(neutral)==len(original)-removed
            variants={'original':original,'headerless':neutral,'common_header':common+neutral}
            for variant,value in variants.items():
                assert 0<len(value)<=cfg['max_length']
                rows[variant].append({'input_ids':value,'attention_mask':[1]*len(value),'labels':0})
            records.append({'task':task,'idx':int(example['idx']),
                            'lengths':{k:len(v) for k,v in variants.items()}})
        truth.extend([source]*len(raw))
        for variant in plan['variants']:predictions[variant].extend(predict(rows[variant]))
        write_json(out/'progress.json',{'state':'running','last_task':task})
    y=np.array(truth);original=np.array(predictions['original']);summary={}
    for variant,values in predictions.items():
        values=np.array(values)
        confusion=np.zeros((len(cfg['tasks']),len(cfg['tasks'])),dtype=int)
        np.add.at(confusion,(y,values),1)
        summary[variant]={'source_accuracy':float((values==y).mean()),
                          'prediction_flip_fraction_vs_original':float((values!=original).mean()),
                          'per_task_accuracy':{t:float((values[y==i]==i).mean()) for i,t in enumerate(cfg['tasks'])},
                          'confusion_true_rows_predicted_columns':confusion.tolist()}
    write_json(out/'summary.json',{'state':'passed','plan':plan,'summary':summary,
               'manifests':manifests,'records':records,'predictions':predictions,
               'true_source':truth,'environment':environment()})
    print('ROUTER_HEADER_DIAGNOSTIC_COMPLETE',out)


if __name__=='__main__':main()
