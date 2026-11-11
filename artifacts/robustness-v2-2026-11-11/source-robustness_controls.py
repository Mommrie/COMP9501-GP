"""Selection-locked lambda/gating controls and cleaned official-validation checks.

Exploratory follow-up: the earlier internal final and historical GLUE validation
have already informed the research. No final-label tuning or blind-test claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import torch

CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'merge_study'))
from common import FIELDS,batches,encode_example,environment,read_config,tokenizer_for,write_json
from independent_protocol import content_key
from activation_extension import pooled,route_values,best_expert
from engine import DeltaEvaluator


def own_routes(cache,tasks):
    return np.concatenate([np.full(len(cache[t]['labels']),i,dtype=int) for i,t in enumerate(tasks)])


def benefit_routes(gains,own,threshold):
    gains=np.asarray(gains,float).copy();own=np.asarray(own,int)
    if gains.shape!=(len(own),4) or not np.isfinite(gains).all():raise ValueError('Invalid predicted gains')
    if threshold is None:return own.copy()
    gains[np.arange(len(own)),own]=-np.inf
    proposed=gains.argmax(1)
    return np.where(gains[np.arange(len(own)),proposed]>threshold,proposed,own)


def confidence_routes(proposed,confidence,own,threshold):
    if threshold is None:return own.copy()
    return np.where(np.asarray(confidence)>=threshold,proposed,own)


def describe(cache,routes,tasks):
    value=route_values(cache,routes,tasks);own=own_routes(cache,tasks)
    acc=[value['tasks'][t]['metrics']['accuracy'] for t in tasks]
    value.update(mean_accuracy=float(np.mean(acc)),worst_accuracy=float(min(acc)),switch_fraction=float((routes!=own).mean()))
    impact={};offset=0
    for i,t in enumerate(tasks):
        c=cache[t];n=len(c['labels']);correct=c['logits'][routes[offset:offset+n],np.arange(n)].argmax(-1)==c['labels']
        old=c['logits'][i].argmax(-1)==c['labels']
        impact[t]={'helps_fraction':float((correct & ~old).mean()),'harms_fraction':float((~correct & old).mean())}
        offset+=n
    value['impact_vs_own']=impact
    return value


def choose_safe(candidates,reference,tolerance):
    eligible=[]
    for i,c in enumerate(candidates):
        v=c['evaluation']
        ok=(v['score']>=reference['score']-1e-12 and v['mean_accuracy']>=reference['mean_accuracy']-1e-12
            and v['worst_accuracy']>=reference['worst_accuracy']-tolerance-1e-12)
        c['selection_eligible']=bool(ok)
        if ok:eligible.append(i)
    if not eligible:raise ValueError('Always-own fallback must be included')
    return max(eligible,key=lambda i:(candidates[i]['evaluation']['score'],candidates[i]['evaluation']['mean_accuracy'],
                                      -candidates[i]['evaluation']['switch_fraction']))


def stored_probabilities(record,features):
    classes=np.asarray(record['classes'],int)
    if len(classes)==1:return np.ones((len(features),1)),classes
    scores=np.asarray(features)@np.asarray(record['coef']).T+np.asarray(record['intercept'])
    if len(classes)==2:
        p=1/(1+np.exp(-np.clip(scores[:,0],-700,700)));return np.stack([1-p,p],1),classes
    scores-=scores.max(1,keepdims=True);p=np.exp(scores);return p/p.sum(1,keepdims=True),classes


def predict_locked_gate(lock,base_features,known_tasks,kind):
    """Reload the selected gate from JSON; always-own needs no base features."""
    own=np.asarray(known_tasks,int)
    if kind not in {'confidence_gate','benefit_gate'}:raise ValueError('Unknown gate')
    selected=lock['confidence_selected' if kind=='confidence_gate' else 'benefit_selected']
    if selected['threshold'] is None:return own.copy()
    x=np.asarray(base_features).copy()
    if len(x)!=len(own) or not np.isfinite(x).all():raise ValueError('Invalid features')
    x-=np.asarray(lock['scaler_mean']);x/=np.asarray(lock['scaler_scale'])
    f=np.concatenate([x,np.eye(4)[own]],1)
    if kind=='confidence_gate':
        probability,classes=stored_probabilities(lock['source_routers']['expert_router'],f)
        i=probability.argmax(1)
        return confidence_routes(classes[i],probability[np.arange(len(i)),i],own,selected['threshold'])
    gains=f@np.asarray(lock['benefit_coef']).T+np.asarray(lock['benefit_intercept'])
    return benefit_routes(gains,own,selected['threshold'])


def clean_indices(rows,fields,train_keys):
    keep=[];keys=[];seen=set();overlap=duplicates=0
    for i,row in enumerate(rows):
        key=content_key([row[f] for f in fields])
        if key in train_keys:overlap+=1;continue
        if key in seen:duplicates+=1;continue
        seen.add(key);keep.append(i);keys.append(key)
    return keep,keys,overlap,duplicates


def load_cache(folder,summary,role,tasks):
    path=folder/(role+'-cache.npz')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==summary['role_cache_sha256'][role]
    with np.load(path,allow_pickle=False) as arrays:
        return {t:{k:arrays[t+'__'+k].copy() for k in ['base','logits','labels']} for t in tasks}


def select_lambdas(previous,current):
    result={};candidates={}
    for family in ['hidden','shuffled']:
        specs=[(0.,'previous','layer_ce_fp32'),(.01,'current','ce_h001' if family=='hidden' else 'ce_s001'),
               (.1,'current','ce_h010' if family=='hidden' else 'ce_s010'),(1.,'previous','layer_ce_hidden' if family=='hidden' else 'layer_ce_shuffled')]
        rows=[]
        for weight,source,name in specs:
            chosen=(previous if source=='previous' else current)['fits'][name]['selected']
            rows.append({'lambda':weight,'source':source,'profile':name,'selection_score':chosen['score'],
                         'weights':chosen['weights'],'gain':chosen['gain'],'step':chosen['step']})
        chosen=max(rows,key=lambda r:(r['selection_score'],-r['lambda']))
        result[family]=chosen;candidates[family]=rows
    return result,candidates


def run_seed(study,previous_dir,current_dir,out,plan,seed):
    from datasets import load_dataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from common import metrics
    from engine import compact
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    cfg=read_config(study/'locked-config.json');tasks=cfg['tasks'];old=previous_dir/f'seed{seed}';fresh=current_dir/f'seed{seed}'
    previous=read_config(old/'summary.json');current=read_config(fresh/'summary.json')
    assert previous['state']==current['state']=='passed'
    assert previous['partition_sha256']==current['partition_sha256']==cfg['partition_manifest_sha256']
    out.mkdir(exist_ok=False);state={'state':'running','seed':seed}
    def publish(stage):state['current']=stage;write_json(out/'status.json',state)
    publish('selection-only')
    choices,lambda_candidates=select_lambdas(previous,current)
    fit=load_cache(fresh,current,'coefficient_fit',tasks);selection=load_cache(fresh,current,'coefficient_selection',tasks)
    reference=describe(selection,own_routes(selection,tasks),tasks)
    scaler=StandardScaler().fit(np.concatenate([fit[t]['base'] for t in tasks]))
    def features(cache):
        own=own_routes(cache,tasks);x=scaler.transform(np.concatenate([cache[t]['base'] for t in tasks]))
        return x,np.concatenate([x,np.eye(4)[own]],1),own
    xf,ff,of=features(fit);xs,fs,os=features(selection)
    rec=read_config(fresh/'selection.json')['routers']
    def expert_proposal(f):
        prob,classes=stored_probabilities(rec['expert_router'],f);i=prob.argmax(1)
        return classes[i],prob[np.arange(len(i)),i]
    proposed,confidence=expert_proposal(fs)
    confidence_candidates=[{'threshold':None,'evaluation':reference}]
    for threshold in plan['confidence_thresholds']:
        if threshold is None:continue
        confidence_candidates.append({'threshold':threshold,'evaluation':describe(selection,confidence_routes(proposed,confidence,os,threshold),tasks)})
    ci=choose_safe(confidence_candidates,reference,plan['selection_worst_accuracy_tolerance'])
    targets=[]
    for i,t in enumerate(tasks):
        correct=fit[t]['logits'].argmax(-1)==fit[t]['labels'][None,:]
        targets.append((correct.astype(float)-correct[i][None,:]).T)
    target=np.concatenate(targets);ridge_candidates=[{'alpha':None,'threshold':None,'evaluation':reference}];models={}
    for alpha in plan['ridge_alphas']:
        model=Ridge(alpha=alpha).fit(ff,target);models[alpha]=model;gain=model.predict(fs)
        for threshold in plan['benefit_thresholds']:
            ridge_candidates.append({'alpha':alpha,'threshold':threshold,'evaluation':describe(selection,benefit_routes(gain,os,threshold),tasks)})
    ri=choose_safe(ridge_candidates,reference,plan['selection_worst_accuracy_tolerance'])
    cr=confidence_candidates[ci];rr=ridge_candidates[ri];ridge=models.get(rr['alpha'])
    baseline=read_config(study/f'weight-baselines/selection-s{seed}.json')['selected']
    lock={'static':choices,'lambda_candidates':lambda_candidates,'confidence_candidates':confidence_candidates,
          'benefit_candidates':ridge_candidates,'confidence_selected':cr,'benefit_selected':rr,
          'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
          'benefit_coef':ridge.coef_.tolist() if ridge is not None else None,'benefit_intercept':ridge.intercept_.tolist() if ridge is not None else None,
          'source_routers':rec,'fixed_baselines':baseline,'locked_before_reading_cached_final_and_loading_official':True,
          'scope':'Internal final and historical official validation already explored. Selection constraints are not final safety guarantees.'}
    write_json(out/'selection-lock.json',lock)
    predictions={};all_results={};dataset_record={}
    def routes(cache):
        x,f,own=features(cache);expert,prob=expert_proposal(f)
        source_p,source_classes=stored_probabilities(rec['source_router'],x)
        assignments={'task_id':own,'source_router':source_classes[source_p.argmax(1)],'expert_router':expert,
                'confidence_gate':confidence_routes(expert,prob,own,cr['threshold']),
                'benefit_gate':benefit_routes(ridge.predict(f) if ridge is not None else np.zeros((len(own),4)),own,rr['threshold']),
                'oracle':np.concatenate([best_expert(cache[t]['logits'],cache[t]['labels']) for t in tasks])}
        for kind in ['confidence_gate','benefit_gate']:
            loaded=predict_locked_gate(lock,np.concatenate([cache[t]['base'] for t in tasks]),own,kind)
            np.testing.assert_array_equal(loaded,assignments[kind])
        return assignments
    def route_evaluate(role,cache):
        r=routes(cache);values={}
        for name,assignment in r.items():
            values[name]=describe(cache,assignment,tasks);predictions[role+'__route__'+name]=assignment
        for t in tasks:
            predictions[role+'__labels__'+t]=cache[t]['labels'];predictions[role+'__expert_logits__'+t]=cache[t]['logits']
        return values
    def score_static(logits,labels):
        per={t:{'metrics':metrics((logits[t],labels[t])),'samples':len(labels[t])} for t in tasks}
        from engine import task_score
        for t in tasks:per[t]['primary_score']=task_score(t,per[t]['metrics'])
        return {'tasks':per,'score':float(np.mean([v['primary_score'] for v in per.values()])),
                'mean_accuracy':float(np.mean([v['metrics']['accuracy'] for v in per.values()])),
                'worst_accuracy':min(v['metrics']['accuracy'] for v in per.values())}
    publish('internal-locked-evaluation')
    final=load_cache(fresh,current,'validation',tasks);all_results['internal']={'routing':route_evaluate('internal',final),'static':{}}
    with np.load(fresh/'predictions.npz',allow_pickle=False) as original:
        for name in ['source_router','expert_router']:
            np.testing.assert_array_equal(predictions['internal__route__'+name],original['route__'+name])
    oldp=np.load(old/'predictions.npz',allow_pickle=False);newp=np.load(fresh/'predictions.npz',allow_pickle=False)
    old_ids=read_config(old/'split-ids.json');new_ids=read_config(fresh/'split-ids.json')
    for t in tasks:
        assert old_ids['validation'][t]==new_ids['validation'][t]
        np.testing.assert_array_equal(oldp['labels__'+t],final[t]['labels'])
        np.testing.assert_array_equal(newp['labels__'+t],final[t]['labels'])
    selected={'ce_fp32':{'source':'previous','profile':'layer_ce_fp32'},'selected_hidden':choices['hidden'],'selected_shuffled':choices['shuffled']}
    for name,choice in selected.items():
        archive=oldp if choice['source']=='previous' else newp
        logits={t:archive[choice['profile']+'__'+t] for t in tasks}
        all_results['internal']['static'][name]=score_static(logits,{t:final[t]['labels'] for t in tasks})
        for t in tasks:predictions['internal__'+name+'__'+t]=logits[t]
    oldp.close();newp.close()
    publish('official-cleaning-after-lock')
    manifest=read_config(study/'partitions.json');train_keys=set()
    for t in tasks:
        raw=load_dataset('nyu-mll/glue',t,split='train',revision=cfg['dataset_revision'])
        assert raw._fingerprint==manifest['tasks'][t]['fingerprint']
        train_keys.update(content_key([row[f] for f in FIELDS[t]]) for row in raw)
    tokenizer=tokenizer_for(cfg);data={}
    for t in tasks:
        raw=load_dataset('nyu-mll/glue',t,split='validation',revision=cfg['dataset_revision'])
        keep,keys,overlap,duplicates=clean_indices(raw,FIELDS[t],train_keys);assert keep
        selected_raw=raw.select(keep)
        dataset_record[t]={'source_split':'validation','source_fingerprint':raw._fingerprint,'raw_count':len(raw),'retained_count':len(keep),
                           'removed_train_tuple_overlap':overlap,'removed_within_duplicates':duplicates,'sample_ids':list(map(int,selected_raw['idx'])),'content_sha256':keys}
        def tokenize(batch,task=t):
            rows=[encode_example(task,cfg,tokenizer,{f:batch[f][i] for f in FIELDS[task]}) for i in range(len(batch['label']))]
            result={k:[row[k] for row in rows] for k in rows[0]};result['labels']=batch['label'];return result
        data[t]=selected_raw.map(tokenize,batched=True,remove_columns=selected_raw.column_names)
    write_json(out/'official-data-manifest.json',{'revision':cfg['dataset_revision'],'tasks':dataset_record,'filter':'normalized full tuples against all original train, then per-task dedup; label-blind filtering','scope':lock['scope']})
    engine=DeltaEvaluator.from_bank(cfg,study/f'bank-s{seed}');zero={k:torch.zeros_like(v) for k,v in engine.banks[tasks[0]].items()};official={}
    for t in tasks:
        publish('official-experts:'+t);engine.set_state(t,zero);features_=[]
        with torch.inference_mode():
            for batch in batches(data[t],tokenizer,cfg['batch_size']):
                batch.pop('labels');inp={k:v.to(engine.device) for k,v in batch.items()}
                response=engine.model(**inp,output_hidden_states=True)
                features_.append(pooled(response.hidden_states[-1],inp['attention_mask']).cpu().numpy())
        logits=[];labels=None
        for expert in tasks:
            value=engine.evaluate(t,engine.banks[expert],data[t],tokenizer,cfg['batch_size']);logits.append(value['logits'])
            if labels is not None:np.testing.assert_array_equal(labels,value['labels'])
            labels=value['labels']
        official[t]={'base':np.concatenate(features_),'logits':np.stack(logits),'labels':labels}
    all_results['official']={'routing':route_evaluate('official',official),'static':{}}
    static_parameters={'ce_fp32':previous['fits']['layer_ce_fp32']['selected'],'selected_hidden':choices['hidden'],'selected_shuffled':choices['shuffled']}
    for name,parameters in list(static_parameters.items())+[(m,baseline[m]) for m in ['average','task_arithmetic','ties','dare']]:
        publish('official-static:'+name)
        if name in baseline:delta=engine.merge(parameters,seed=42)
        else:
            w=np.asarray(parameters['weights']);gain=parameters['gain']
            delta={k:sum(engine.banks[t][k]*float(w[0 if len(w)==1 else int(k.split('.')[2]),i])*gain for i,t in enumerate(tasks)) for k in engine.base}
        logits={};labels={}
        for t in tasks:
            value=engine.evaluate(t,delta,data[t],tokenizer,cfg['batch_size']);logits[t]=value['logits'];labels[t]=value['labels']
            np.testing.assert_array_equal(labels[t],official[t]['labels'])
            predictions['official__'+name+'__'+t]=logits[t]
        all_results['official']['static'][name]=score_static(logits,labels)
    np.savez_compressed(out/'predictions.npz',**predictions)
    versions=environment()
    from importlib.metadata import version
    versions['packages'].update({name:version(name) for name in ['numpy','scikit-learn','scipy']})
    summary={'state':'passed','seed':seed,'results':all_results,'environment':versions,'partition_sha256':cfg['partition_manifest_sha256'],
             'plan':plan,'selection':lock,'official_counts':{t:dataset_record[t]['retained_count'] for t in tasks}}
    write_json(out/'summary.json',summary);state.update(state='passed');publish(None)
    del engine;torch.cuda.empty_cache()


def main():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Run model evaluation in Colab')
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--previous',required=True);p.add_argument('--sensitivity',required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);plan=read_config(args.plan)
    if plan['lambda_grid']!=[0.,.01,.1,1.]:raise ValueError('This frozen selector requires the declared four-point lambda grid')
    if any(t<0 for t in plan['benefit_thresholds']) or any(a<=0 for a in plan['ridge_alphas']):raise ValueError('Invalid conservative-routing grid')
    write_json(out/'plan.json',plan);write_json(out/'source-sha256.json',{str(p.relative_to(CODE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),CODE/'activation_extension.py',CODE/'common.py',CODE/'merge_study/engine.py']})
    state={'state':'running','completed_seeds':[]}
    try:
        for seed in plan['seeds']:
            state['current_seed']=seed;write_json(out/'status.json',state)
            run_seed(Path(args.study),Path(args.previous),Path(args.sensitivity),out/f'seed{seed}',plan,seed)
            state['completed_seeds'].append(seed);write_json(out/'status.json',state)
        state.update(state='passed',current_seed=None);write_json(out/'status.json',state)
    except Exception as error:
        state.update(state='error',error=str(error),traceback=traceback.format_exc());write_json(out/'status.json',state);raise


if __name__=='__main__':main()
