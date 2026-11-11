"""Regularized Fisher/LDA and task-ID residual gates on frozen CSA features."""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
from summarize_csa_bridge import project,native

ROOT=Path(__file__).resolve().parents[1]
TASKS=['sst2','mrpc','rte','cola']


def fit_lda(x,y,shrinkage):
    """Empirical-prior shared-covariance LDA; Woodbury avoids a large D x D solve."""
    x=np.asarray(x,float);y=np.asarray(y);classes,inverse,counts=np.unique(y,return_inverse=True,return_counts=True)
    if not 0<shrinkage<=1 or len(x)!=len(y) or not np.isfinite(x).all():raise ValueError('Invalid LDA data or shrinkage')
    means=np.stack([x[inverse==i].mean(0) for i in range(len(classes))]);residual=x-means[inverse]
    variance=float(np.square(residual).sum()/len(x)/x.shape[1]);diagonal=max(shrinkage*variance,1e-8)
    factor=residual*np.sqrt((1-shrinkage)/len(x))
    if x.shape[1]>len(x):
        # (a I + A.T A)^-1 M.T = (M.T - A.T(a I + A A.T)^-1 A M.T)/a.
        weights=(means.T-factor.T@np.linalg.solve(factor@factor.T+diagonal*np.eye(len(x)),factor@means.T))/diagonal
    else:weights=np.linalg.solve(factor.T@factor+diagonal*np.eye(x.shape[1]),means.T)
    intercept=-.5*np.sum(means.T*weights,axis=0)+np.log(counts/len(x))
    return {'classes':classes,'coef':weights.T,'intercept':intercept}


def probabilities(model,x):
    values=x@model['coef'].T+model['intercept'];values-=values.max(1,keepdims=True)
    values=np.exp(values);return values/values.sum(1,keepdims=True)


def expert_prob(model,x):
    result=np.zeros((len(x),4));result[:,model['classes'].astype(int)]=probabilities(model,x);return result


def prior_routes(prob,own,margin):
    if margin is None:return own.copy()
    other=prob.copy();other[np.arange(len(own)),own]=-np.inf;proposed=other.argmax(1)
    return np.where(other[np.arange(len(own)),proposed]-prob[np.arange(len(own)),own]>margin,proposed,own)


def gain_routes(gains,own,threshold):
    if threshold is None:return own.copy()
    other=gains.copy();other[np.arange(len(own)),own]=-np.inf;proposed=other.argmax(1)
    return np.where(other[np.arange(len(own)),proposed]>threshold,proposed,own)


def choose(candidates,reference,tolerance,safe):
    eligible=[]
    for i,c in enumerate(candidates):
        v=c['metrics'];ok=not safe or (v['native']>=reference['native']-1e-12 and v['accuracy']>=reference['accuracy']-1e-12 and v['worst_accuracy']>=reference['worst_accuracy']-tolerance-1e-12)
        c['eligible']=bool(ok)
        if ok:eligible.append(i)
    if not eligible:raise ValueError('Missing always-own fallback')
    return max(eligible,key=lambda i:(candidates[i]['metrics']['native'],candidates[i]['metrics']['accuracy'] if safe else 0.,
         -candidates[i]['metrics']['switch_fraction'] if safe else 0.,candidates[i]['alpha'],candidates[i].get('threshold',-1) or 0.))


def own_ids(cache):return np.concatenate([np.full(len(cache[t+'__labels']),i,int) for i,t in enumerate(TASKS)])


def describe(cache,routes):
    score,per=native(cache,routes,TASKS);offset=0;helps=[];harms=[];own=own_ids(cache)
    for i,t in enumerate(TASKS):
        y=cache[t+'__labels'];n=len(y);pred=cache[t+'__logits'].argmax(-1)
        a=pred[routes[offset:offset+n],np.arange(n)]==y;b=pred[i]==y
        helps.append(np.mean(a&~b));harms.append(np.mean(~a&b));offset+=n
    return {'native':score,'accuracy':float(np.mean([v['accuracy'] for v in per.values()])),
            'worst_accuracy':float(min(v['accuracy'] for v in per.values())),'switch_fraction':float(np.mean(routes!=own)),
            'helps':float(np.mean(helps)),'harms':float(np.mean(harms)),'tasks':per}


def best_labels(cache):
    result=[]
    for t in TASKS:
        values=cache[t+'__logits'];values=values-values.max(-1,keepdims=True)
        lp=values-np.log(np.exp(values).sum(-1,keepdims=True));y=cache[t+'__labels']
        result.append(lp[:,np.arange(len(y)),y].argmax(0))
    return np.concatenate(result)


def benefit_labels(cache):
    result=[]
    for i,t in enumerate(TASKS):
        y=cache[t+'__labels'];correct=cache[t+'__logits'].argmax(-1)==y[None,:]
        result.append((correct.astype(int)-correct[i].astype(int)).T)
    return np.concatenate(result)


def transform(params,key,cache,response):
    return np.concatenate([cache[t+'__base'] for t in TASKS]) if key=='base' else project(params,key,response)


def run_seed(bridge,previous,out,plan,seed):
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);out.mkdir(exist_ok=False);started=time.perf_counter()
    def save(name,value):(out/name).write_text(json.dumps(value,indent=2))
    params=np.load(bridge/f'seed{seed}/fitted-parameters.npz');responses=np.load(bridge/f'seed{seed}/response-cache.npz')
    fit=np.load(previous/f'seed{seed}/coefficient_fit-cache.npz');sel=np.load(previous/f'seed{seed}/coefficient_selection-cache.npz')
    ids=json.loads((bridge/f'seed{seed}/split-ids.json').read_text());oldids=json.loads((previous/f'seed{seed}/split-ids.json').read_text());assert ids==oldids
    previous_summary=json.loads((bridge/f'seed{seed}/summary.json').read_text());keys=list(previous_summary['selection'])
    ownfit=own_ids(fit);ownsel=own_ids(sel);target=best_labels(fit);benefit=benefit_labels(fit);reference=describe(sel,ownsel)
    state={'state':'running','seed':seed};save('status.json',state);locks={};models={};scalers={};arrays={}
    for key in keys:
        state['current']=key;save('status.json',state)
        xf=transform(params,key,fit,responses['coefficient_fit']);xs=transform(params,key,sel,responses['coefficient_selection'])
        scale=StandardScaler().fit(xf);scalers[key]=scale;arrays[key+'__mean']=scale.mean_;arrays[key+'__scale']=scale.scale_
        a=np.column_stack([scale.transform(xf),np.eye(4)[ownfit]]);b=np.column_stack([scale.transform(xs),np.eye(4)[ownsel]])
        candidates={family:[] for family in ['direct_lda','task_prior','benefit_lda']}
        for alpha in plan['lda_shrinkage']:
            model=fit_lda(a,target,alpha);models[(key,alpha,'expert')]=model;prob=expert_prob(model,b)
            candidates['direct_lda'].append({'alpha':alpha,'threshold':-1,'metrics':describe(sel,prob.argmax(1))})
            for margin in plan['prior_margins']:
                candidates['task_prior'].append({'alpha':alpha,'threshold':margin,'metrics':describe(sel,prior_routes(prob,ownsel,margin))})
            gains=[]
            for expert in range(4):
                mask=ownfit!=expert;model=fit_lda(a[mask],benefit[mask,expert],alpha);models[(key,alpha,'benefit'+str(expert))]=model
                gains.append(probabilities(model,b)@model['classes'])
            gains=np.column_stack(gains)
            for threshold in plan['benefit_thresholds']:
                candidates['benefit_lda'].append({'alpha':alpha,'threshold':threshold,'metrics':describe(sel,gain_routes(gains,ownsel,threshold))})
        for family in ['task_prior','benefit_lda']:
            candidates[family].append({'alpha':1.,'threshold':None,'always_own':True,'metrics':reference})
        locks[key]={}
        for family,options in candidates.items():
            selected=choose(options,reference,plan['worst_accuracy_tolerance'],family!='direct_lda')
            locks[key][family]={'selected':selected,'candidates':options}
    # Save all candidate discriminants to allow independent selection as well as final audit.
    for (key,alpha,target_name),model in models.items():
        prefix=key+'__'+str(alpha)+'__'+target_name
        for name,value in model.items():arrays[prefix+'__'+name]=value
    np.savez_compressed(out/'fitted-discriminants.npz',**arrays);save('selection-lock.json',locks)
    # Read only after every representation/family has locked its choices.
    final=np.load(previous/f'seed{seed}/validation-cache.npz');response=responses['validation'];own=own_ids(final)
    predictions={};results={}
    for key in keys:
        raw=transform(params,key,final,response);a=np.column_stack([scalers[key].transform(raw),np.eye(4)[own]])
        results[key]={}
        for family,record in locks[key].items():
            selected=record['candidates'][record['selected']];alpha=selected['alpha'];threshold=selected['threshold']
            if selected.get('always_own'):routes=own.copy()
            elif family in ['direct_lda','task_prior']:
                prob=expert_prob(models[(key,alpha,'expert')],a)
                routes=prob.argmax(1) if family=='direct_lda' else prior_routes(prob,own,threshold)
            else:
                gains=np.column_stack([probabilities(models[(key,alpha,'benefit'+str(j))],a)@models[(key,alpha,'benefit'+str(j))]['classes'] for j in range(4)])
                routes=gain_routes(gains,own,threshold)
            results[key][family]=describe(final,routes);predictions[key+'__'+family+'__routes']=routes
    results['task_id']={'reference':describe(final,own)};predictions['task_id__routes']=own
    for t in TASKS:
        predictions[t+'__labels']=final[t+'__labels'];predictions[t+'__logits']=final[t+'__logits']
    np.savez_compressed(out/'predictions.npz',**predictions)
    save('summary.json',{'seed':seed,'methods':results,'seconds':time.perf_counter()-started,'selection_reference':reference})
    save('split-ids.json',ids);state.update(state='passed',current=None);save('status.json',state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bridge',required=True);p.add_argument('--previous',required=True);p.add_argument('--output',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);plan=json.loads(Path(a.plan).read_text())
    def save(name,v):(out/name).write_text(json.dumps(v,indent=2))
    save('plan.json',plan);save('source-sha256.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'code/summarize_csa_bridge.py']})
    state={'state':'running','completed_seeds':[]};save('status.json',state)
    try:
        for seed in plan['seeds']:
            state['current_seed']=seed;save('status.json',state)
            run_seed(Path(a.bridge),Path(a.previous),out/f'seed{seed}',plan,seed);state['completed_seeds'].append(seed);save('status.json',state)
        state.update(state='passed',current_seed=None);save('status.json',state)
    except Exception as e:
        state.update(state='error',error=str(e),traceback=traceback.format_exc());save('status.json',state);raise


if __name__=='__main__':main()
