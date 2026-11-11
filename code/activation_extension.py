"""Exploratory hidden-response and best-expert controls on the locked v3 bank."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'merge_study'))
import numpy as np
import torch
from common import batches,dataset_for,dataset_manifest,environment,metrics,model_for,read_config,tokenizer_for,write_json
from coefficient_merge.core import Coefficients,MixedLinear
from engine import DeltaEvaluator,compact,task_score


def pooled(hidden,mask):
    mask=mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden.float()*mask).sum(1)/mask.sum(1).clamp_min(1)


def hidden_loss(actual,target,floor):
    return ((actual.float()-target.float()).square().mean(-1)/target.float().square().mean(-1).clamp_min(floor)).mean()


def profile_hidden_weight(plan,profile):
    value=float(profile.get('hidden_weight',plan['hidden_weight'])) if profile.get('hidden',True) else 0.
    if not np.isfinite(value) or value<0:raise ValueError('Hidden weight must be finite and nonnegative')
    return value


def derangement(n,seed):
    if n<2:raise ValueError('At least two examples are required')
    rng=np.random.default_rng(seed)
    for _ in range(10000):
        p=rng.permutation(n)
        if np.all(p!=np.arange(n)):return p
    raise RuntimeError('Could not generate fixed derangement')


def best_expert(logits,labels):
    # logits: expert x sample x class; choose highest true-class probability.
    shifted=logits-logits.max(-1,keepdims=True)
    logp=shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))
    return np.stack([-logp[i,np.arange(len(labels)),labels] for i in range(len(logits))]).argmin(0)


def route_values(cache,assign,tasks):
    values={};offset=0
    for t in tasks:
        c=cache[t];n=len(c['labels']);r=np.asarray(assign[offset:offset+n],dtype=int)
        logits=c['logits'][r,np.arange(n)]
        score=metrics((logits,c['labels']))
        values[t]={'metrics':score,'primary_score':task_score(t,score),'samples':n}
        offset+=n
    return {'tasks':values,'score':sum(v['primary_score'] for v in values.values())/len(tasks)}


def run_seed(study,out,plan,seed):
    from safetensors.torch import load_file
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    cfg=read_config(study/'locked-config.json');tasks=cfg['tasks'];bank=study/f'bank-s{seed}'
    out.mkdir(exist_ok=False)
    status={'state':'running','seed':seed,'plan':plan,'fits':{},'selection':{},'final_static':{},'final_router':{}}
    started=time.perf_counter()
    def publish(current):
        status.update(current=current,wall_seconds=time.perf_counter()-started)
        write_json(out/'status.tmp',status);os.replace(out/'status.tmp',out/'status.json')
    try:
        tokenizer=tokenizer_for(cfg);engine=DeltaEvaluator.from_bank(cfg,bank)
        zero={k:torch.zeros_like(v) for k,v in engine.banks[tasks[0]].items()}
        data={};cache={};ids={}
        @torch.inference_mode()
        def collect(task,delta,rows):
            engine.set_state(task,delta);hs=[];ls=[];ys=[]
            for batch in batches(rows,tokenizer,cfg['batch_size']):
                ys.append(batch.pop('labels').numpy());inp={k:v.to(engine.device) for k,v in batch.items()}
                result=engine.model(**inp,output_hidden_states=True)
                hs.append(pooled(result.hidden_states[-1],inp['attention_mask']).cpu().numpy())
                ls.append(result.logits.float().cpu().numpy())
            hidden,logits,labels=np.concatenate(hs),np.concatenate(ls),np.concatenate(ys)
            if not np.isfinite(hidden).all() or not np.isfinite(logits).all():raise ValueError('Non-finite cached response')
            return hidden,logits,labels
        def load_role(role):
            data[role]={t:dataset_for(t,cfg,tokenizer,role,0) for t in tasks};cache[role]={};ids[role]={}
            for t in tasks:
                publish('cache:'+role+':'+t)
                rows=data[role][t];base,_,labels=collect(t,zero,rows);hs=[];ls=[]
                for expert in tasks:
                    h,l,y=collect(t,engine.banks[expert],rows);assert np.array_equal(y,labels);hs.append(h);ls.append(l)
                cache[role][t]={'base':base,'hidden':np.stack(hs),'logits':np.stack(ls),'labels':labels}
                ids[role][t]=dataset_manifest(t,cfg,role,0)['sample_ids']
            if plan.get('save_role_cache',False):
                path=out/(role+'-cache.npz')
                np.savez_compressed(path,**{t+'__'+key:cache[role][t][key] for t in tasks for key in ['base','logits','labels']})
                status.setdefault('role_cache_sha256',{})[role]=hashlib.sha256(path.read_bytes()).hexdigest()
        load_role('coefficient_fit');load_role('coefficient_selection')
        for t in tasks:
            teacher=read_config(bank/t/'run.json')
            assert teacher['config']['partition_manifest_sha256']==cfg['partition_manifest_sha256']
            prior=read_config(bank/t/'data-manifest.json')
            a=set(prior['train']['sample_ids'])|set(prior['selection']['sample_ids'])
            assert a.isdisjoint(ids['coefficient_fit'][t]+ids['coefficient_selection'][t])
        targets={t:torch.from_numpy(cache['coefficient_fit'][t]['hidden'][i]-cache['coefficient_fit'][t]['base']) for i,t in enumerate(tasks)}
        basefit={t:torch.from_numpy(cache['coefficient_fit'][t]['base']) for t in tasks}
        permutations={t:derangement(len(targets[t]),seed+19000+i) for i,t in enumerate(tasks)}
        write_json(out/'fit-control.json',{'permutations':{t:p.tolist() for t,p in permutations.items()},
            'target_sha256':{t:hashlib.sha256(v.numpy().tobytes()).hexdigest() for t,v in targets.items()},
            'target_delta_energy':{t:{'mean':float(v.square().mean(-1).mean()),
                'minimum':float(v.square().mean(-1).min()),
                'floor_fraction':float((v.square().mean(-1)<plan['hidden_energy_floor']).float().mean())} for t,v in targets.items()}})
        torch.save({'base':basefit,'target_delta':targets},out/'fit-targets.pt')
        def delta(weights,gain):
            return {k:sum(engine.banks[t][k]*float(weights[0 if len(weights)==1 else int(k.split('.')[2]),i])*gain for i,t in enumerate(tasks)) for k in engine.base}
        def assess_delta(d,rows,predictions=None,name=None):
            result={}
            for t in tasks:
                r=engine.evaluate(t,d,rows[t],tokenizer,cfg['batch_size']);result[t]=compact(r)
                if predictions is not None:predictions[name+'__'+t]=r['logits']
            return {'tasks':result,'score':sum(v['primary_score'] for v in result.values())/len(tasks)}
        factors={}
        for t in tasks:
            raw=load_file(str(bank/t/'adapter/adapter_model.safetensors'));factors[t]={}
            for k,a in raw.items():
                if k.endswith('.lora_A.weight'):
                    stem=k[:-len('.lora_A.weight')];factors[t][stem.removeprefix('base_model.model.')]=(a.float().to(engine.device),raw[stem+'.lora_B.weight'].float().to(engine.device))
        for profile in plan['profiles']:
            name=profile['name'];publish('fit:'+name);student=model_for(cfg).to(engine.device)
            for p in student.parameters():p.requires_grad_(False)
            coeff=Coefficients(profile['layers']).to(engine.device)
            for module_name in factors[tasks[0]]:
                old=student.get_submodule(module_name);parent,child=module_name.rsplit('.',1)
                layer=0 if profile['layers']==1 else int(module_name.split('.')[2])
                setattr(student.get_submodule(parent),child,MixedLinear(old,[factors[t][module_name] for t in tasks],coeff,layer,cfg['alpha']/cfg['rank']))
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            t=tasks[0];batch=next(iter(batches(data['coefficient_fit'][t].select(range(8)),tokenizer,8)));batch.pop('labels')
            inp={k:v.to(engine.device) for k,v in batch.items()};student.eval()
            with torch.inference_mode():
                student.score.weight.copy_(engine.heads[t].to(student.score.weight));engine.set_state(t,delta(torch.full((1,4),.25),2.))
                torch.testing.assert_close(student(**inp).logits,engine.model(**inp).logits,atol=1e-3,rtol=1e-4)
            optimizer=torch.optim.Adam(coeff.parameters(),lr=plan['learning_rate']);generator=torch.Generator().manual_seed(seed+9000)
            best=None;history=[];losses=[]
            def assess(step):
                nonlocal best
                w=coeff.weights().detach().cpu();g=float(coeff.gain().detach())
                value=assess_delta(delta(w,g),data['coefficient_selection']);value.update(step=step,weights=w.tolist(),gain=g);history.append(value)
                if best is None or value['score']>best['score']:best=copy.deepcopy(value)
            assess(0)
            for step in range(1,plan['steps']+1):
                t=tasks[(step-1)%len(tasks)];idx=torch.randint(len(targets[t]),(plan['batch_size'],),generator=generator).tolist()
                batch=next(iter(batches(data['coefficient_fit'][t].select(idx),tokenizer,plan['batch_size'])));labels=batch.pop('labels').to(engine.device)
                inp={k:v.to(engine.device) for k,v in batch.items()}
                targetidx=permutations[t][idx].tolist() if profile['shuffle'] else idx
                target=targets[t][targetidx].to(engine.device);base=basefit[t][idx].to(engine.device)
                with torch.no_grad():student.score.weight.copy_(engine.heads[t].to(student.score.weight))
                student.train();optimizer.zero_grad(set_to_none=True)
                # Same FP32 student precision for the CE, hidden, and shuffled controls.
                with torch.autocast(device_type='cuda',enabled=False):
                    result=student(**inp,output_hidden_states=True)
                    response=pooled(result.hidden_states[-1],inp['attention_mask'])-base
                    hm=hidden_loss(response,target,plan['hidden_energy_floor'])
                    ce=torch.nn.functional.cross_entropy(result.logits.float(),labels) if profile['ce'] else hm.new_zeros(())
                    loss=ce+profile_hidden_weight(plan,profile)*hm+plan['penalty']*coeff.penalty()
                if not torch.isfinite(loss):raise ValueError('Non-finite objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(coeff.parameters(),1.)
                if step==1:
                    gradient=float(coeff.logits.grad.norm())+float(coeff.gain_logit.grad.abs());assert gradient>1e-9
                    allowed={id(p) for p in coeff.parameters()};assert all(p.grad is None for p in student.parameters() if id(p) not in allowed)
                optimizer.step();losses.append({'step':step,'hidden':float(hm.detach()),'ce':float(ce.detach()),'total':float(loss.detach())})
                if step in plan['check_steps']:publish('select:'+name+':'+str(step));assess(step)
            status['fits'][name]={'selected':best,'history':history,'losses':losses,'parameters':sum(p.numel() for p in coeff.parameters()),'first_gradient':gradient}
            status['selection'][name]={'weights':best['weights'],'gain':best['gain'],'step':best['step']}
            del student,optimizer,coeff;torch.cuda.empty_cache()
        publish('fit:routers')
        fitcache=cache['coefficient_fit'];selcache=cache['coefficient_selection']
        xf=np.concatenate([fitcache[t]['base'] for t in tasks]);xs=np.concatenate([selcache[t]['base'] for t in tasks])
        yf=np.concatenate([np.full(len(fitcache[t]['labels']),i) for i,t in enumerate(tasks)])
        ys=np.concatenate([np.full(len(selcache[t]['labels']),i) for i,t in enumerate(tasks)])
        bestfit=np.concatenate([best_expert(fitcache[t]['logits'],fitcache[t]['labels']) for t in tasks])
        scaler=StandardScaler().fit(xf);af=scaler.transform(xf);ass=scaler.transform(xs)
        models={};router_records={}
        for kind,labels in [('source_router',yf),('expert_router',bestfit)]:
            a=np.concatenate([af,np.eye(len(tasks))[yf]],1) if kind=='expert_router' else af
            b=np.concatenate([ass,np.eye(len(tasks))[ys]],1) if kind=='expert_router' else ass
            candidates=[];fitted=[]
            for C in plan['router_C']:
                model=(DummyClassifier(strategy='most_frequent') if len(np.unique(labels))<2 else LogisticRegression(C=C,max_iter=plan['router_max_iter'],random_state=seed)).fit(a,labels)
                routes=model.predict(b).astype(int);value=route_values(selcache,routes,tasks)
                candidates.append({'C':C,'score':value['score'],'source_accuracy':float((routes==ys).mean())});fitted.append(model)
            chosen=int(np.argmax([v['score'] for v in candidates]));models[kind]=fitted[chosen]
            model=models[kind];router_records[kind]={'candidates':candidates,'selected':candidates[chosen],
                'classes':model.classes_.tolist(),'coef':model.coef_.tolist() if hasattr(model,'coef_') else None,
                'intercept':model.intercept_.tolist() if hasattr(model,'intercept_') else None}
        status['router_selection']=router_records
        write_json(out/'selection.json',{'static':status['selection'],'routers':router_records,'scaler_mean':scaler.mean_.tolist(),
            'scaler_scale':scaler.scale_.tolist(),'best_expert_fit_targets':bestfit.tolist(),'locked_before_final_load':True,
            'scope':'Exploratory extension; previous v3 final already inspected; no new hyperparameter selection on final.'})
        publish('final:load-after-lock');load_role('validation');final=data['validation'];fc=cache['validation'];predictions={}
        for name,value in status['selection'].items():
            publish('final:static:'+name)
            d=delta(torch.tensor(value['weights']),value['gain'])
            status['final_static'][name]=assess_delta(d,final,predictions,name)
            diagnostic={}
            for t in tasks:
                h,_,_=collect(t,d,data['coefficient_fit'][t])
                diagnostic[t]=float(hidden_loss(torch.from_numpy(h)-basefit[t],targets[t],plan['hidden_energy_floor']))
            status['final_static'][name]['aligned_fit_hidden_mse']=diagnostic
        xv=np.concatenate([fc[t]['base'] for t in tasks]);yv=np.concatenate([np.full(len(fc[t]['labels']),i) for i,t in enumerate(tasks)])
        av=scaler.transform(xv);routes={'task_id':yv.copy(),'oracle':np.concatenate([best_expert(fc[t]['logits'],fc[t]['labels']) for t in tasks]),
            'random':np.random.default_rng(seed+29000).integers(0,len(tasks),len(yv))}
        for kind,model in models.items():routes[kind]=model.predict(np.concatenate([av,np.eye(len(tasks))[yv]],1) if kind=='expert_router' else av).astype(int)
        for name,assignment in routes.items():
            status['final_router'][name]=route_values(fc,assignment,tasks)
            status['final_router'][name]['source_accuracy']=float((assignment==yv).mean())
            status['final_router'][name]['oracle_agreement']=float((assignment==routes['oracle']).mean())
            impact={};offset=0
            for i,t in enumerate(tasks):
                c=fc[t];n=len(c['labels']);chosen=c['logits'][assignment[offset:offset+n],np.arange(n)].argmax(-1)==c['labels']
                own=c['logits'][i].argmax(-1)==c['labels']
                impact[t]={'helps_fraction':float((chosen & ~own).mean()),'harms_fraction':float((~chosen & own).mean())}
                offset+=n
            status['final_router'][name]['impact_vs_own']=impact
            predictions['route__'+name]=assignment
        old=read_config(study/f'coefficients-s{seed}/summary.json')
        for t in tasks:
            observed=status['final_router']['task_id']['tasks'][t]['metrics'];historical=old['individual'][t]['metrics']
            status.setdefault('historical_task_id_parity',{})[t]={'exact_metrics':observed==historical,
                'accuracy_delta':observed['accuracy']-historical['accuracy']}
            if abs(observed['accuracy']-historical['accuracy'])>1/len(fc[t]['labels']):
                raise ValueError('Historical teacher accuracy differs by more than one example; investigate runtime or restore')
            assert status['final_router']['oracle']['tasks'][t]['metrics']['accuracy']>=old['individual'][t]['metrics']['accuracy']
            predictions['labels__'+t]=fc[t]['labels'];predictions['expert_logits__'+t]=fc[t]['logits']
        np.savez_compressed(out/'predictions.npz',**predictions)
        write_json(out/'split-ids.json',ids)
        status.update(state='passed',environment=environment(),partition_sha256=cfg['partition_manifest_sha256'])
        publish(None);write_json(out/'summary.json',status)
    except Exception as error:
        status.update(state='error',error=str(error),traceback=traceback.format_exc());publish(status.get('current'));raise


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<20*2**30:
        raise RuntimeError('Run on a Colab CUDA GPU with at least 20 GiB VRAM (A100 or L4)')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True);p.add_argument('--plan',required=True);args=p.parse_args()
    study,out=Path(args.study),Path(args.output);out.mkdir(parents=True,exist_ok=False);plan=read_config(args.plan)
    if plan['student_precision']!='float32':raise ValueError('This controlled extension requires float32')
    write_json(out/'plan.json',plan);write_json(out/'source-sha256.json',{str(p.relative_to(CODE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),CODE/'common.py',CODE/'coefficient_merge/core.py',CODE/'merge_study/engine.py']})
    state={'state':'running','completed_seeds':[],'plan':plan};write_json(out/'status.json',state)
    try:
        for seed in plan['seeds']:
            state['current_seed']=seed;write_json(out/'status.json',state);run_seed(study,out/f'seed{seed}',plan,seed)
            state['completed_seeds'].append(seed);write_json(out/'status.json',state)
        state.update(state='passed',current_seed=None);write_json(out/'status.json',state)
    except Exception as error:
        state.update(state='error',error=str(error));write_json(out/'status.json',state);raise


if __name__=='__main__':main()
