"""Task-conditioned linear-response matching in the LoRA span."""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'merge_study'))
import torch
from common import read_config,write_json,tokenizer_for,dataset_for,dataset_manifest,batches,environment
from engine import DeltaEvaluator,compact
from maths import solve


def response_grams(engine,cfg,tokenizer,bank,limit,progress):
    from safetensors.torch import load_file
    tasks=cfg['tasks'];factors={}
    for task in tasks:
        state=load_file(str(bank/task/'adapter/adapter_model.safetensors'))
        factors[task]={}
        for key,a in state.items():
            if key.endswith('.lora_A.weight'):
                stem=key[:-len('.lora_A.weight')]
                name=stem.removeprefix('base_model.model.')
                factors[task][name]=(a.to(engine.device),state[stem+'.lora_B.weight'].to(engine.device))
    names=sorted(factors[tasks[0]])
    layers=sorted({int(n.split('.')[2]) for n in names})
    grams=torch.zeros(len(tasks),len(layers),4,4,dtype=torch.float64)
    context={}
    for teacher,task in enumerate(tasks):
        progress('responses:'+task)
        accum={n:torch.zeros(4,4,device=engine.device,dtype=torch.float64) for n in names}
        counts={n:0 for n in names};hooks=[]
        def hook(name):
            def receive(module,inputs):
                x=inputs[0][context['mask']]
                ys=[]
                for t in tasks:
                    a,b=factors[t][name]
                    ys.append(torch.nn.functional.linear(torch.nn.functional.linear(x,a),b)*(cfg['alpha']/cfg['rank']))
                y=torch.stack(ys).flatten(1)
                accum[name].add_((y@y.T).double()/ys[0].shape[-1])
                counts[name]+=len(x)
            return receive
        for name in names:hooks.append(engine.model.get_submodule(name).register_forward_pre_hook(hook(name)))
        engine.set_state(task,engine.banks[task])
        data=dataset_for(task,cfg,tokenizer,'calibration',limit)
        try:
            with torch.inference_mode():
                for batch in batches(data,tokenizer,cfg['batch_size']):
                    batch.pop('labels');inputs={k:v.to(engine.device) for k,v in batch.items()}
                    context['mask']=inputs['attention_mask'].bool()
                    engine.model(**inputs)
        finally:
            for handle in hooks:handle.remove()
        for name in names:
            layer=layers.index(int(name.split('.')[2]))
            grams[teacher,layer]+=accum[name].cpu()/counts[name]
    return grams,layers


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--bank',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    cfg,plan=read_config(args.config),read_config(args.plan);bank=Path(args.bank);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(4)
    prior=read_config(bank/'status.json');assert prior['state']=='passed' and prior['config']==cfg
    status={'state':'running','seed':prior['seed'],'config':cfg,'plan':plan,'calibration':{},'validation':{},'coefficients':{},
            'source_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')}}
    start=time.perf_counter()
    def publish(current):
        status.update(current=current,wall_seconds=time.perf_counter()-start)
        write_json(out/'status.tmp',status);os.replace(out/'status.tmp',out/'status.json')
    try:
        publish('load-engine');tokenizer=tokenizer_for(cfg);engine=DeltaEvaluator.from_bank(cfg,bank)
        tasks=cfg['tasks'];gram,layers=response_grams(engine,cfg,tokenizer,bank,plan['stats_samples_per_task'],publish)
        torch.save({'grams':gram,'layers':layers,'task_order':tasks},out/'response-grams.pt')
        G=gram.sum(0);b=sum(gram[i,:,:,i] for i in range(4))
        coefficients={'uniform':torch.full((len(layers),4),.25,dtype=torch.float64)}
        for mode in ['global','layer']:
            for beta in plan['betas']:
                name=f'{mode}-b{beta}'
                if mode=='global':
                    scales=G.diagonal(dim1=-2,dim2=-1).sum(-1).clamp_min(1e-12)
                    a,info=solve((G/scales[:,None,None]).sum(0),(b/scales[:,None]).sum(0),beta)
                    coefficients[name]=a.repeat(len(layers),1);status['coefficients'][name]={'weights':coefficients[name].tolist(),'solver':[info]}
                else:
                    solved=[solve(g,v,beta) for g,v in zip(G,b)]
                    coefficients[name]=torch.stack([x[0] for x in solved]);status['coefficients'][name]={'weights':coefficients[name].tolist(),'solver':[x[1] for x in solved]}
        def merge(name,gain):
            a=coefficients[name]
            return {key:sum(engine.banks[t][key]*float(a[layers.index(int(key.split('.')[2])),i])*gain for i,t in enumerate(tasks)) for key in engine.base}
        calibration={t:dataset_for(t,cfg,tokenizer,'calibration',plan['selection_samples_per_task']) for t in tasks}
        manifests={t:dataset_manifest(t,cfg,'calibration',plan['selection_samples_per_task']) for t in tasks}
        for task in tasks:
            trained=read_config(bank/task/'data-manifest.json')
            assert set(manifests[task]['sample_ids']).isdisjoint(trained['train']['sample_ids'])
        write_json(out/'calibration-manifests.json',manifests)
        specs=[]
        for name in coefficients:
            for gain in plan['gains']:
                tag=f'{name}-g{gain}';spec={'name':name,'gain':gain,'tag':tag};specs.append(spec)
                delta=merge(name,gain);values={}
                for task in tasks:
                    publish('calibration:'+tag+':'+task)
                    values[task]=compact(engine.evaluate(task,delta,calibration[task],tokenizer,cfg['batch_size']))
                status['calibration'][tag]={'spec':spec,'tasks':values,'score':sum(v['primary_score'] for v in values.values())/4}
                del delta
        selection={mode:max([sp for sp in specs if sp['name'].startswith(mode)],key=lambda sp:status['calibration'][sp['tag']]['score']) for mode in plan['modes']}
        status['selection']={'selected':selection,'locked_before_validation':True,'plan_sha256':hashlib.sha256(Path(args.plan).read_bytes()).hexdigest()}
        write_json(out/'selection.json',status['selection']);publish('load-validation-after-lock')
        data={t:dataset_for(t,cfg,tokenizer,'validation',0) for t in tasks}
        for mode,spec in selection.items():
            delta=merge(spec['name'],spec['gain']);values={}
            for task in tasks:
                publish('validation:'+mode+':'+task);values[task]=compact(engine.evaluate(task,delta,data[task],tokenizer,cfg['batch_size']))
            status['validation'][mode]={'spec':spec,'tasks':values,'score':sum(v['primary_score'] for v in values.values())/4}
            del delta
        status.update(state='passed',environment=environment());publish(None);write_json(out/'summary.json',status)
        print('FUNCTIONAL_MERGE_COMPLETE',out,flush=True)
    except Exception as error:
        status.update(state='error',error=str(error),traceback=traceback.format_exc());publish(status.get('current'));raise


if __name__=='__main__':main()
