import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'merge_study'))
import numpy as np
import torch
from common import read_config,write_json,tokenizer_for,model_for,dataset_for,dataset_manifest,batches,metrics,environment
from engine import DeltaEvaluator,compact,task_score
from core import Coefficients,MixedLinear


def main():
    from safetensors.torch import load_file
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--output',required=True);p.add_argument('--config',required=True);p.add_argument('--plan',required=True);args=p.parse_args()
    cfg,plan=read_config(args.config),read_config(args.plan);bank=Path(args.bank);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(4);prior=read_config(bank/'status.json');seed=prior['seed'];tasks=cfg['tasks']
    status={'state':'running','seed':seed,'config':cfg,'plan':plan,'fits':{},'validation':{},'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}}
    start=time.perf_counter()
    def publish(current):
        status.update(current=current,wall_seconds=time.perf_counter()-start);write_json(out/'status.tmp',status);os.replace(out/'status.tmp',out/'status.json')
    try:
        publish('data');tokenizer=tokenizer_for(cfg);fit={};select={};ids={}
        for t in tasks:
            data=dataset_for(t,cfg,tokenizer,'calibration',plan['total_calibration_samples']);split=plan['fit_samples_per_task']
            fit[t]=data.select(range(split));select[t]=data.select(range(split,len(data)))
            m=dataset_manifest(t,cfg,'calibration',plan['total_calibration_samples'])
            ids[t]={'fit':m['sample_ids'][:split],'selection':m['sample_ids'][split:]}
            train_ids=set(read_config(bank/t/'data-manifest.json')['train']['sample_ids'])
            assert set(ids[t]['fit']).isdisjoint(ids[t]['selection']) and train_ids.isdisjoint(m['sample_ids'])
        write_json(out/'split-ids.json',ids);publish('teachers');engine=DeltaEvaluator.from_bank(cfg,bank)
        teachers={t:torch.from_numpy(engine.evaluate(t,engine.banks[t],fit[t],tokenizer,cfg['batch_size'])['logits']) for t in tasks}
        torch.save(teachers,out/'teacher-logits.pt')
        def delta(weights,gain):
            return {key:sum(engine.banks[t][key]*float(weights[0 if len(weights)==1 else int(key.split('.')[2]),i])*gain for i,t in enumerate(tasks)) for key in engine.base}
        def evaluate_delta(d,data):
            values={t:compact(engine.evaluate(t,d,data[t],tokenizer,cfg['batch_size'])) for t in tasks}
            return {'tasks':values,'score':sum(v['primary_score'] for v in values.values())/4}
        uniform={}
        for gain in plan['uniform_gains']:
            publish('uniform-selection:'+str(gain));uniform[str(gain)]=evaluate_delta(delta(torch.full((1,4),.25),gain),select)
        best_gain=max(plan['uniform_gains'],key=lambda g:uniform[str(g)]['score']);status['uniform_selection']=uniform
        saved={'uniform':{'weights':[[.25]*4],'gain':best_gain}}
        factors={}
        for t in tasks:
            raw=load_file(str(bank/t/'adapter/adapter_model.safetensors'));factors[t]={}
            for k,a in raw.items():
                if k.endswith('.lora_A.weight'):
                    stem=k[:-len('.lora_A.weight')];factors[t][stem.removeprefix('base_model.model.')]=(a.to(engine.device),raw[stem+'.lora_B.weight'].to(engine.device))
        for profile in plan['profiles']:
            name=profile['name'];publish('fit:'+name);student=model_for(cfg).to(engine.device)
            for parameter in student.parameters():parameter.requires_grad_(False)
            coeff=Coefficients(profile['layers']).to(engine.device)
            for module_name in factors[tasks[0]]:
                old=student.get_submodule(module_name);parent_name,child=module_name.rsplit('.',1)
                layer=0 if profile['layers']==1 else int(module_name.split('.')[2])
                setattr(student.get_submodule(parent_name),child,MixedLinear(old,[factors[t][module_name] for t in tasks],coeff,layer,cfg['alpha']/cfg['rank']))
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            student.eval();t=tasks[0];batch=next(iter(batches(fit[t].select(range(8)),tokenizer,8)));batch.pop('labels')
            inputs={k:v.to(engine.device) for k,v in batch.items()}
            with torch.inference_mode():
                student.score.weight.copy_(engine.heads[t].to(student.score.weight));engine.set_state(t,delta(torch.full((1,4),.25),2.))
                left=student(**inputs).logits;right=engine.model(**inputs).logits
                torch.testing.assert_close(left,right,atol=1e-3,rtol=1e-4)
                parity=float((left-right).abs().max())
            optimizer=torch.optim.Adam(coeff.parameters(),lr=plan['learning_rate']);generator=torch.Generator().manual_seed(seed+9000)
            best=None;history=[]
            def assess(step):
                nonlocal best
                w=coeff.weights().detach().cpu();gain=float(coeff.gain().detach())
                value=evaluate_delta(delta(w,gain),select);value.update(step=step,gain=gain,weights=w.tolist());history.append(value)
                if best is None or value['score']>best['score']:best=copy.deepcopy(value)
            assess(0)
            first_gradient=None
            for step in range(1,plan['steps']+1):
                t=tasks[(step-1)%4];idx=torch.randint(len(fit[t]),(plan['batch_size'],),generator=generator).tolist()
                batch=next(iter(batches(fit[t].select(idx),tokenizer,plan['batch_size'])));labels=batch.pop('labels').to(engine.device)
                inputs={k:v.to(engine.device) for k,v in batch.items()}
                with torch.no_grad():student.score.weight.copy_(engine.heads[t].to(student.score.weight))
                student.train();optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=engine.device=='cuda'):
                    logits=student(**inputs).logits.float()
                    if profile['objective']=='ce':loss=torch.nn.functional.cross_entropy(logits,labels)
                    else:
                        T=plan['temperature'];target=teachers[t][idx].to(engine.device)
                        loss=torch.nn.functional.kl_div((logits/T).log_softmax(-1),(target/T).softmax(-1),reduction='batchmean')*T*T
                    loss=loss+plan['penalty']*coeff.penalty()
                loss.backward();torch.nn.utils.clip_grad_norm_(coeff.parameters(),1.)
                if step==1:
                    first_gradient=float(coeff.logits.grad.norm())+float(coeff.gain_logit.grad.abs());assert first_gradient>1e-9
                    assert all(parameter.grad is None for parameter in student.parameters() if id(parameter) not in {id(v) for v in coeff.parameters()})
                optimizer.step()
                if step in plan['check_steps']:publish('select:'+name+':'+str(step));assess(step)
            saved[name]={'weights':best['weights'],'gain':best['gain'],'step':best['step']}
            status['fits'][name]={'history':history,'selected':best,'first_coefficient_gradient':first_gradient,'initial_logit_error':parity,'learnable_parameters':sum(v.numel() for v in coeff.parameters())}
            del student,optimizer,coeff
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        write_json(out/'selection.json',{'selected':saved,'locked_before_validation':True,'plan_sha256':hashlib.sha256(Path(args.plan).read_bytes()).hexdigest()});status['selection']=saved
        publish('validation-after-lock');data={t:dataset_for(t,cfg,tokenizer,'validation',0) for t in tasks}
        for name,value in saved.items():
            publish('validation:'+name);status['validation'][name]=evaluate_delta(delta(torch.tensor(value['weights']),value['gain']),data)
        status.update(state='passed',environment=environment());publish(None);write_json(out/'summary.json',status)
        print('COEFFICIENT_MERGE_COMPLETE',out)
    except Exception as error:
        status.update(state='error',error=str(error),traceback=traceback.format_exc());publish(status.get('current'));raise


if __name__=='__main__':main()
