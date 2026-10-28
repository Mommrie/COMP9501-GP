"""Replay a fixed coefficient history with an explicitly label-free criterion."""
from pathlib import Path
import sys,json,os,time,hashlib
CODE=Path(__file__).resolve().parents[1];ROOT=CODE.parent
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'merge_study'))
import torch,numpy as np
from common import read_config,write_json,tokenizer_for,dataset_for,batches,environment
from engine import DeltaEvaluator,compact


def main():
 plan=read_config(ROOT/'configs/selection-control.json');cfg=read_config(ROOT/'configs/protocol-v2-a100.json')
 out=ROOT/'outputs/selection-control';out.mkdir(exist_ok=False);torch.set_num_threads(4)
 status={'state':'running','plan':plan,'runs':{},'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}};start=time.perf_counter()
 def publish(current):
  status.update(current=current,seconds=time.perf_counter()-start);write_json(out/'status.tmp',status);os.replace(out/'status.tmp',out/'status.json')
 publish('start')
 for seed in plan['seeds']:
  bank=ROOT/('outputs/bank-v2-a100-s42' if seed==42 else f'outputs/replication-s43-s44/bank-s{seed}')
  prior=read_config(ROOT/f'outputs/coefficient-v1/s{seed}/summary.json');engine=DeltaEvaluator.from_bank(cfg,bank);tok=tokenizer_for(cfg);tasks=cfg['tasks']
  selection={t:dataset_for(t,cfg,tok,'calibration',plan['total_calibration_samples']).select(range(plan['fit_samples_per_task'],min(plan['total_calibration_samples'],len(dataset_for(t,cfg,tok,'calibration',plan['total_calibration_samples']))))) for t in tasks}
  def logits(task,d,data):
   engine.set_state(task,d);values=[]
   with torch.inference_mode():
    for batch in batches(data,tok,cfg['batch_size']):
     batch.pop('labels')  # Do not compute a label-dependent metric during selection.
     values.append(engine.model(**{k:v.to(engine.device) for k,v in batch.items()}).logits.float().cpu())
   value=torch.cat(values);assert torch.isfinite(value).all();return value
  def delta(value):
   weights=torch.tensor(value['weights']);gain=value['gain']
   return {key:sum(engine.banks[t][key]*float(weights[0 if len(weights)==1 else int(key.split('.')[2]),i])*gain for i,t in enumerate(tasks)) for key in engine.base}
  teacher={t:logits(t,engine.banks[t],selection[t]) for t in tasks};T=plan['temperature']
  def assess(value):
   losses={}
   for t in tasks:
    student=logits(t,delta(value),selection[t])
    losses[t]=float(torch.nn.functional.kl_div((student/T).log_softmax(-1),(teacher[t]/T).softmax(-1),reduction='batchmean')*T*T)
    assert losses[t]>=-1e-6
   return {'tasks':losses,'mean_kl':sum(losses.values())/len(tasks)}
  candidates={};selected={};r={'candidates':candidates,'selected':selected,'validation':{},'selection_samples':{t:len(d) for t,d in selection.items()}}
  candidates['uniform']=[]
  for gain in plan['uniform_gains']:
   v={'weights':[[.25]*4],'gain':gain};publish(f's{seed}:uniform:{gain}');candidates['uniform'].append({'value':v,**assess(v)})
  selected['uniform']=min(candidates['uniform'],key=lambda v:v['mean_kl'])['value']
  for profile in plan['profiles']:
   candidates[profile]=[]
   history=prior['fits'][profile]['history'];assert [v['step'] for v in history]==plan['steps']
   for h in history:
    v={k:h[k] for k in ['weights','gain','step']};publish(f's{seed}:{profile}:{h["step"]}');candidates[profile].append({'value':v,**assess(v)})
   selected[profile]=min(candidates[profile],key=lambda v:v['mean_kl'])['value']
  write_json(out/f's{seed}-locked.json',{'selected':selected,'locked_before_validation':True})
  validation={t:dataset_for(t,cfg,tok,'validation',0) for t in tasks}
  for name,v in selected.items():
   publish(f's{seed}:validation:{name}');values={t:compact(engine.evaluate(t,delta(v),validation[t],tok,cfg['batch_size'])) for t in tasks}
   r['validation'][name]={'tasks':values,'score':sum(v['primary_score'] for v in values.values())/len(tasks),'gt_selected_score':prior['validation'][name]['score'],'gt_selected_step':prior['selection'][name].get('step'),'gt_selected_gain':prior['selection'][name]['gain']}
  status['runs'][str(seed)]=r;publish(f's{seed}:passed');del engine;torch.cuda.empty_cache()
 status.update(state='passed',environment=environment());publish(None);write_json(out/'summary.json',status);print('SELECTION_CONTROL_PASSED')


if __name__=='__main__':main()
