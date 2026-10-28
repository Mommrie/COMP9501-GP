from pathlib import Path
import json
from statistics import mean,stdev
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2]
s=json.loads((root/'artifacts/transfer-check-2026-10-25.json').read_text())
result={};rows=[]
for method in s['plan']['methods']:
 result[method]={}
 for dataset in ['imdb','paws']:
  result[method][dataset]={metric:{'mean':mean(r[method][dataset]['metrics'][metric] for r in s['runs'].values()),'sample_sd':stdev(r[method][dataset]['metrics'][metric] for r in s['runs'].values())} for metric in ['accuracy','f1','mcc']}
 a=result[method];rows.append('| '+method+' | '+' | '.join(f"{a[d][m]['mean']*scale:.2f} +/- {a[d][m]['sample_sd']*scale:.2f}" for d,m,scale in [('imdb','accuracy',100),('paws','accuracy',100),('paws','f1',1)])+' |')
(root/'artifacts/transfer-check-aggregate-2026-10-25.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
methods=s['plan']['methods'];names=['Individual','Average','Uniform gain 2','Layer CE (GT sel.)','Layer KL (KL sel.)'];x=np.arange(len(names))
for ax,dataset in zip(axes,['imdb','paws']):
 vals=[result[m][dataset]['accuracy'] for m in methods]
 ax.bar(x,[v['mean']*100 for v in vals],yerr=[v['sample_sd']*100 for v in vals],capsize=4,color=['#73808a','#8c9ea4','#b18751','#21618c','#c45436'])
 prior=s['datasets'][dataset]['positive_fraction'];ax.axhline(max(prior,1-prior)*100,color='black',linestyle='--',label='Subset majority baseline')
 ax.set_xticks(x);ax.set_xticklabels(names,rotation=25,ha='right');ax.set_ylim(0,100);ax.set_title(dataset.upper()+': 512 label-blind test examples');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8,loc='upper right')
axes[0].set_ylabel('Accuracy (%), mean +/- sample SD over 3 adapter seeds')
fig.suptitle('Fixed transfer checks: no target adaptation or method selection\nSource task heads and 128-token input protocol retained')
fig.savefig(root/'artifacts/transfer-check-2026-10-25.png',dpi=180);plt.close(fig)
print('| Method | IMDb acc % | PAWS acc % | PAWS F1 |\n|---|---:|---:|---:|\n'+'\n'.join(rows))
