"""Archive fixed-plan descriptive statistics and a standalone scientific plot."""
from pathlib import Path
import json
from statistics import mean,stdev
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
s=json.loads((root/'artifacts/summary-2026-10-25.json').read_text())
groups={};lines=['| Train / digit | Core size | CSA 1-NN (%) | PCA 1-NN (%) | CSA test MSE | PCA test MSE |','|---:|---:|---:|---:|---:|---:|']
for rank in s['plan']['ranks']:
 for n in s['plan']['train_per_class']:
  runs=[r for r in s['runs'] if r['n_per_class']==n and r['rank']==rank]
  if not runs:continue
  g={}
  for name in ['CSA','PCA']:
   g[name]={k:{'mean':mean(r['methods'][name][k] for r in runs),'sd':stdev(r['methods'][name][k] for r in runs)} for k in ['accuracy_1nn','train_mse','test_mse','fit_seconds']}
  groups[f'n{n}-r{rank}']=g
  values=[]
  for metric,mul,fmt in [('accuracy_1nn',100,'.1f'),('test_mse',1,'.4f')]:
   for name in ['CSA','PCA']:
    v=g[name][metric];values.append(f"{v['mean']*mul:{fmt}} +/- {v['sd']*mul:{fmt}}")
  lines.append('| '+ ' | '.join([str(n),str(rank*rank),*values])+' |')
(root/'artifacts/aggregate-2026-10-25.json').write_text(json.dumps(groups,indent=2)+'\n')
(root/'artifacts/table-2026-10-25.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for col,rank in enumerate([2,4]):
 ns=[n for n in s['plan']['train_per_class'] if f'n{n}-r{rank}' in groups]
 for name,color in [('CSA','#21618c'),('PCA','#c45436')]:
  gs=[groups[f'n{n}-r{rank}'][name] for n in ns]
  axes[0,col].errorbar(ns,[g['accuracy_1nn']['mean']*100 for g in gs],yerr=[g['accuracy_1nn']['sd']*100 for g in gs],marker='o',capsize=3,label=name,color=color)
  axes[1,col].errorbar(ns,[g['test_mse']['mean'] for g in gs],yerr=[g['test_mse']['sd'] for g in gs],marker='o',capsize=3,label=name,color=color)
 axes[0,col].set_title(f'Matched latent dimension {rank*rank}; basis CSA {28*rank} vs PCA {196*rank*rank}')
 for row in range(2):
  axes[row,col].set_xscale('log');axes[row,col].set_xticks(ns);axes[row,col].set_xticklabels(ns);axes[row,col].grid(alpha=.25);axes[row,col].legend();axes[row,col].set_xlabel('Training images per digit')
axes[0,0].set_ylabel('Held-out 1-NN accuracy (%)');axes[1,0].set_ylabel('Held-out pixel MSE (inputs in [0,1])')
fig.suptitle('2D CSA illustration on 14x14 MNIST: mean +/- sample SD of three nested-subset seeds\nNot the original face/Gabor-MNIST experiments; equal core size is not equal total storage')
fig.savefig(root/'artifacts/csa-comparison-2026-10-25.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
for seed in [21,22,23]:
 r=next(r for r in s['runs'] if r['seed']==seed and r['rank']==4 and r['n_per_class']==3)
 h=np.array(r['methods']['CSA']['history']);ax.plot(np.arange(len(h)),h/h[-1],marker='.',label=f'Subset seed {seed}')
ax.set_xlabel('Coordinate update (0 = initialization)');ax.set_ylabel('Captured energy / final energy');ax.set_title('Alternating updates increase captured training energy\nConvergence of objective does not establish a global optimum');ax.legend();ax.grid(alpha=.2)
fig.savefig(root/'artifacts/csa-convergence-2026-10-25.png',dpi=180);plt.close(fig)
print('\n'.join(lines));print('CSA_REPORT_READY')
