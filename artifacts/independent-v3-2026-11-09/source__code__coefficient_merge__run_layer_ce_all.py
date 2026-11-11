import os
from pathlib import Path
import sys
import subprocess
import json

root=Path(__file__).resolve().parents[2]
out=root/'outputs/coefficient-ce-control'
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True)
state={'state':'running','runs':{}}
def publish():
    (out/'status.tmp').write_text(json.dumps(state,indent=2))
    os.replace(out/'status.tmp',out/'status.json')
publish()
for seed in [42,43,44]:
    bank=root/('outputs/bank-v2-a100-s42' if seed==42 else f'outputs/replication-s43-s44/bank-s{seed}')
    state['current']=seed;publish()
    with (out/f's{seed}.log').open('w') as log:
        r=subprocess.run([sys.executable,'code/coefficient_merge/run.py','--config','configs/protocol-v2-a100.json','--plan','configs/coefficient-ce-control.json','--bank',str(bank),'--output',str(out/f's{seed}')],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:
        state.update(state='error',error=f'seed {seed} return code {r.returncode}');publish();raise RuntimeError(state['error'])
    state['runs'][str(seed)]=json.loads((out/f's{seed}/summary.json').read_text());publish()
state.update(state='passed',current=None);publish()
