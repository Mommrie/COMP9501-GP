"""Apply the unchanged Part 1 TensorCSA to real LoRA response matrices."""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'part1-reproduction-20261025/code'))
from csa_tensor import TensorCSA


def storage(shape,ranks,n,kind):
    e,h=shape;d=int(np.prod(ranks))
    basis=e*ranks[0]+h*ranks[1] if kind=='csa' else e*h*d
    return {'mean':e*h,'basis':int(basis),'codes':n*d,'total':int(e*h+basis+n*d),'raw':n*e*h,'latent':d}


def budget_pca_rank(shape,ranks,n):
    s=storage(shape,ranks,n,'csa');return (s['total']-s['mean'])//(n+int(np.prod(shape)))


class FlatPCA:
    def __init__(self,rank):self.rank=rank
    def fit(self,x):
        self.mean=x.mean(0);z=(x-self.mean).reshape(len(x),-1)
        if not 1<=self.rank<=min(z.shape[1],len(z)-1):raise ValueError('Unsupported PCA rank')
        _,_,v=np.linalg.svd(z,full_matrices=False);self.basis=v[:self.rank].T
        return self
    def transform(self,x):return (x-self.mean).reshape(len(x),-1)@self.basis
    def reconstruct(self,x):return (self.transform(x)@self.basis.T).reshape(x.shape)+self.mean


def relative_error(x,reconstructed,mean):
    return float(np.square(x-reconstructed).sum()/max(np.square(x-mean).sum(),1e-30))


def run_seed(study,previous,out,plan,seed):
    import torch
    from threadpoolctl import threadpool_limits
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from activation_extension import pooled,best_expert,route_values
    from common import read_config,tokenizer_for,dataset_for,dataset_manifest,batches,write_json
    sys.path.insert(0,str(ROOT/'code/merge_study'))
    from engine import DeltaEvaluator
    threadpool_limits(4);torch.set_num_threads(4)
    out.mkdir(exist_ok=False);started=time.perf_counter();status={'state':'running','seed':seed}
    def progress(stage):
        status.update(stage=stage,seconds=time.perf_counter()-started);write_json(out/'status.json',status)
    cfg=read_config(study/'locked-config.json');tasks=cfg['tasks'];tok=tokenizer_for(cfg)
    engine=DeltaEvaluator.from_bank(cfg,study/f'bank-s{seed}');zero={k:torch.zeros_like(v) for k,v in engine.banks[tasks[0]].items()}
    responses={};cache={};ids={}
    @torch.inference_mode()
    def collect(task,delta,rows):
        engine.set_state(task,delta);hidden=[];logits=[];labels=[]
        for batch in batches(rows,tok,cfg['batch_size']):
            labels.append(batch.pop('labels').numpy());inp={k:v.to(engine.device) for k,v in batch.items()}
            result=engine.model(**inp,output_hidden_states=True)
            hidden.append(pooled(result.hidden_states[-1],inp['attention_mask']).cpu().numpy());logits.append(result.logits.float().cpu().numpy())
        return np.concatenate(hidden),np.concatenate(logits),np.concatenate(labels)
    def load(role):
        progress('collect:'+role);xs=[];cache[role]={};ids[role]={}
        old=np.load(previous/f'seed{seed}'/(role+'-cache.npz'))
        for task in tasks:
            progress('collect:'+role+':'+task);rows=dataset_for(task,cfg,tok,role,0)
            base,_,labels=collect(task,zero,rows);hs=[];ls=[]
            assert np.array_equal(labels,old[task+'__labels'])
            for expert in tasks:
                h,l,y=collect(task,engine.banks[expert],rows);assert np.array_equal(y,labels);hs.append(h-base);ls.append(l)
            logits=np.stack(ls);assert np.allclose(logits,old[task+'__logits'],atol=2e-5,rtol=2e-5)
            xs.append(np.stack(hs,axis=1));cache[role][task]={'logits':logits,'labels':labels,'base':base}
            ids[role][task]=dataset_manifest(task,cfg,role,0)['sample_ids']
        responses[role]=np.concatenate(xs).astype(np.float64)
    # Crucially, final responses and labels are not collected until all C choices lock.
    load('coefficient_fit');load('coefficient_selection')
    fit=responses['coefficient_fit'];selection=responses['coefficient_selection'];shape=fit.shape[1:];n=plan['storage_examples']
    assert len(fit)==n
    models={};representations={};records={};pca_ranks=set()
    for ranks in plan['ranks']:
        pca_ranks.update([int(np.prod(ranks)),budget_pca_rank(shape,ranks,n)])
    progress('fit:shared-PCA-SVD')
    # One shared training-only SVD supplies all nested PCA ranks.
    mean=fit.mean(0);_,_,vectors=np.linalg.svd((fit-mean).reshape(len(fit),-1),full_matrices=False)
    for rank in sorted(pca_ranks):
        model=FlatPCA(rank);model.mean=mean;model.basis=vectors[:rank].T;models['pca'+str(rank)]=model
    for ranks in plan['ranks']:
        key='csa'+str(ranks[0])+'x'+str(ranks[1]);progress('fit:'+key)
        models[key]=TensorCSA(ranks,max_sweeps=plan['max_sweeps'],tol=plan['tolerance']).fit(fit)
    for key,model in models.items():
        kind='csa' if key.startswith('csa') else 'pca';ranks=model.ranks if kind=='csa' else [1,model.rank]
        representations[key]=(model.transform(fit).reshape(len(fit),-1),model.transform(selection).reshape(len(selection),-1))
        records[key]={'kind':kind,'ranks':list(ranks),'storage':storage(shape,ranks,n,kind),
                      'fit_error':relative_error(fit,model.reconstruct(fit),model.mean),
                      'selection_error':relative_error(selection,model.reconstruct(selection),model.mean)}
        if kind=='csa':
            records[key].update(history=model.history,converged=model.converged,sweeps=model.sweeps,
                                matched_pca='pca'+str(int(np.prod(ranks))),budget_pca='pca'+str(budget_pca_rank(shape,ranks,n)))
            assert records[records[key]['matched_pca']]['fit_error']<=records[key]['fit_error']+1e-8
    representations['raw_delta']=(fit.reshape(len(fit),-1),selection.reshape(len(selection),-1))
    representations['base']=(np.concatenate([cache['coefficient_fit'][t]['base'] for t in tasks]),np.concatenate([cache['coefficient_selection'][t]['base'] for t in tasks]))
    def own(role):return np.concatenate([np.full(len(cache[role][t]['labels']),i) for i,t in enumerate(tasks)])
    def add_task(x,role):return np.concatenate([x,np.eye(len(tasks))[own(role)]],axis=1)
    target=np.concatenate([best_expert(cache['coefficient_fit'][t]['logits'],cache['coefficient_fit'][t]['labels']) for t in tasks])
    routers={};locked={}
    for key,(a,b) in representations.items():
        progress('router:'+key);scale=StandardScaler().fit(a);a=add_task(scale.transform(a),'coefficient_fit');b=add_task(scale.transform(b),'coefficient_selection')
        candidates=[];fitted=[]
        for c in plan['router_C']:
            model=LogisticRegression(C=c,max_iter=3000,random_state=seed) if len(np.unique(target))>1 else DummyClassifier(strategy='most_frequent')
            model.fit(a,target);value=route_values(cache['coefficient_selection'],model.predict(b),tasks)
            candidates.append({'C':c,'score':value['score']});fitted.append(model)
        best=max(range(len(candidates)),key=lambda i:(candidates[i]['score'],-candidates[i]['C']))
        routers[key]=(scale,fitted[best]);locked[key]={'selected':candidates[best],'candidates':candidates}
    write_json(out/'selection-lock.json',locked);write_json(out/'split-ids.json',ids)
    # Persist fitted representation and router parameters for independent reload audit.
    params={}
    for key,model in models.items():
        params[key+'__mean']=model.mean
        if key.startswith('csa'):
            for i,f in enumerate(model.factors):params[key+'__factor'+str(i)]=f
        else:params[key+'__basis']=model.basis
    for key,(scale,model) in routers.items():
        params[key+'__scale_mean']=scale.mean_;params[key+'__scale_scale']=scale.scale_;params[key+'__classes']=model.classes_
        if hasattr(model,'coef_'):params[key+'__coef']=model.coef_;params[key+'__intercept']=model.intercept_
    np.savez_compressed(out/'fitted-parameters.npz',**params)
    load('validation');final=responses['validation'];predictions={};final_scores={}
    for key,model in models.items():records[key]['final_error']=relative_error(final,model.reconstruct(final),model.mean)
    final_features={key:model.transform(final).reshape(len(final),-1) for key,model in models.items()}
    final_features['raw_delta']=final.reshape(len(final),-1);final_features['base']=np.concatenate([cache['validation'][t]['base'] for t in tasks])
    for key,features in final_features.items():
        scale,model=routers[key];routes=model.predict(add_task(scale.transform(features),'validation')).astype(int)
        final_scores[key]=route_values(cache['validation'],routes,tasks);predictions[key+'__routes']=routes
    routes=own('validation');final_scores['task_id']=route_values(cache['validation'],routes,tasks);predictions['task_id__routes']=routes
    oracle=np.concatenate([best_expert(cache['validation'][t]['logits'],cache['validation'][t]['labels']) for t in tasks])
    final_scores['oracle_GT']=route_values(cache['validation'],oracle,tasks);predictions['oracle_GT__routes']=oracle
    for t in tasks:
        predictions[t+'__labels']=cache['validation'][t]['labels'];predictions[t+'__logits']=cache['validation'][t]['logits']
    np.savez_compressed(out/'predictions.npz',**predictions)
    # Raw response caches are backed up separately; publication contains derived evidence only.
    np.savez_compressed(out/'response-cache.npz',**responses)
    write_json(out/'split-ids.json',ids)
    summary={'seed':seed,'shape':list(shape),'counts':{k:len(v) for k,v in responses.items()},'representations':records,'routing':final_scores,'selection':locked,'seconds':time.perf_counter()-started}
    write_json(out/'summary.json',summary);status['state']='passed';progress('complete')
    del engine;torch.cuda.empty_cache()


def main():
    import torch
    from common import write_json
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--previous',required=True);p.add_argument('--output',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Run in Colab CUDA')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);plan=json.loads(Path(a.plan).read_text())
    write_json(out/'plan.json',plan);write_json(out/'source-sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'part1-reproduction-20261025/code/csa_tensor.py',ROOT/'code/activation_extension.py']})
    state={'state':'running','completed_seeds':[]};write_json(out/'status.json',state)
    try:
        for seed in plan['seeds']:
            state['current_seed']=seed;write_json(out/'status.json',state)
            run_seed(Path(a.study),Path(a.previous),out/f'seed{seed}',plan,seed)
            state['completed_seeds'].append(seed);write_json(out/'status.json',state)
        state.update(state='passed',current_seed=None);write_json(out/'status.json',state)
    except Exception as e:
        state.update(state='error',error=str(e),traceback=traceback.format_exc());write_json(out/'status.json',state);raise


if __name__=='__main__':main()
