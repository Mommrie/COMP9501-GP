from pathlib import Path
import sys,hashlib,json,os,time
CODE=Path(__file__).resolve().parents[1];ROOT=CODE.parent
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'merge_study'))
import numpy as np
import torch
from datasets import load_dataset,Dataset
from huggingface_hub import HfApi
from common import read_config,write_json,tokenizer_for,encode_example,FIELDS,environment
from engine import DeltaEvaluator,compact


def canonical(values):
 return tuple(sorted(' '.join(s.lower().split()) for s in values))


def main():
 plan=read_config(ROOT/'configs/transfer-check.json');cfg=read_config(ROOT/'configs/protocol-v2-a100.json');out=ROOT/'outputs/transfer-check'
 out.mkdir(exist_ok=False);torch.set_num_threads(4);tok=tokenizer_for(cfg)
 status={'state':'running','plan':plan,'datasets':{},'runs':{},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};start=time.perf_counter()
 def publish(current):
  status.update(current=current,seconds=time.perf_counter()-start);write_json(out/'status.tmp',status);os.replace(out/'status.tmp',out/'status.json')
 publish('datasets');data={}
 for spec in plan['datasets']:
  task=spec['head'];revision=HfApi().dataset_info(spec['repo']).sha
  raw=load_dataset(spec['repo'],*([spec['config']] if spec['config'] else []),split=spec['split'],revision=revision)
  known=set()
  for split in ['train','validation']:
   source=load_dataset('nyu-mll/glue',task,split=split,revision=cfg.get('dataset_revision'))
   known.update(canonical([row[f] for f in FIELDS[task]]) for row in source)
  allowed=[i for i,row in enumerate(raw) if canonical([row[f] for f in spec['source_fields']]) not in known]
  rng=np.random.default_rng(plan['sample_seed']);ids=np.array(allowed)[rng.permutation(len(allowed))[:plan['samples_per_dataset']]].tolist()
  rows=[];lengths=[];digests=[]
  for i in ids:
   row=raw[i];values=[row[f] for f in spec['source_fields']];example=dict(zip(FIELDS[task],values))
   encoded=encode_example(task,cfg,tok,example);encoded['labels']=int(row['label']);assert encoded['labels'] in [0,1];rows.append(encoded)
   if len(values)==1:full=len(tok('Text: '+values[0])['input_ids'])
   else:full=len(tok('Sentence A: '+values[0]+'\nSentence B: '+values[1])['input_ids'])
   lengths.append(full);digests.append(hashlib.sha256(json.dumps(canonical(values),ensure_ascii=False).encode()).hexdigest())
  data[spec['name']]=Dataset.from_list(rows)
  status['datasets'][spec['name']]={'spec':spec,'revision':revision,'fingerprint':raw._fingerprint,'native_test_size':len(raw),'exact_duplicates_excluded':len(raw)-len(allowed),'selected_native_row_indices':ids,'selected_text_sha256':digests,'labels':[r['labels'] for r in rows],'positive_fraction':float(np.mean([r['labels'] for r in rows])),'full_token_length_mean':float(np.mean(lengths)),'truncated_fraction':float(np.mean(np.array(lengths)>cfg['max_length']))}
  publish('dataset:'+spec['name'])
 write_json(out/'target-manifest.json',status['datasets'])
 for seed in plan['seeds']:
  bank=ROOT/('outputs/bank-v2-a100-s42' if seed==42 else f'outputs/replication-s43-s44/bank-s{seed}')
  engine=DeltaEvaluator.from_bank(cfg,bank)
  ce=read_config(ROOT/f'outputs/coefficient-ce-control/s{seed}/selection.json')['selected']['layer_ce']
  kl=read_config(ROOT/f'outputs/selection-control/s{seed}-locked.json')['selected']['layer_kl']
  def delta(v):
   w=torch.tensor(v['weights']);return {key:sum(engine.banks[t][key]*float(w[0 if len(w)==1 else int(key.split('.')[2]),i])*v['gain'] for i,t in enumerate(cfg['tasks'])) for key in engine.base}
  choices={'average':{'weights':[[.25]*4],'gain':1},'uniform_gain2':{'weights':[[.25]*4],'gain':2},'layer_ce_gt':ce,'layer_kl_teacher_selected':kl}
  result={};status['runs'][str(seed)]=result
  for method in plan['methods']:
   result[method]={}
   for spec in plan['datasets']:
    publish(f's{seed}:{method}:{spec["name"]}');t=spec['head'];d=engine.banks[t] if method=='individual' else delta(choices[method])
    result[method][spec['name']]=compact(engine.evaluate(t,d,data[spec['name']],tok,cfg['batch_size']))
   result[method]['mean_accuracy']=sum(result[method][s['name']]['metrics']['accuracy'] for s in plan['datasets'])/len(plan['datasets'])
  del engine;torch.cuda.empty_cache();publish(f's{seed}:passed')
 status.update(state='passed',environment=environment());publish(None);write_json(out/'summary.json',status);print('TRANSFER_CHECK_PASSED')


if __name__=='__main__':main()
