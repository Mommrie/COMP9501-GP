"""Independent projector/router reload audit and publication of the CSA bridge."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def native(cache,routes,tasks):
    from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef
    offset=0;values={}
    for t in tasks:
        y=cache[t+'__labels'];n=len(y);pred=cache[t+'__logits'][routes[offset:offset+n],np.arange(n)].argmax(-1)
        acc=accuracy_score(y,pred);f1=f1_score(y,pred,zero_division=0);mcc=matthews_corrcoef(y,pred)
        values[t]={'accuracy':acc,'f1':f1,'mcc':mcc,'primary':mcc if t=='cola' else (acc+f1)/2 if t=='mrpc' else acc}
        offset+=n
    return float(np.mean([v['primary'] for v in values.values()])),values


def project(params,key,x):
    if key=='raw_delta':return x.reshape(len(x),-1)
    z=x-params[key+'__mean']
    if key.startswith('csa'):
        return np.einsum('ea,neh,hb->nab',params[key+'__factor0'],z,params[key+'__factor1']).reshape(len(x),-1)
    return z.reshape(len(x),-1)@params[key+'__basis']


def reconstruct(params,key,x):
    core=project(params,key,x)
    if key.startswith('csa'):
        u,v=params[key+'__factor0'],params[key+'__factor1']
        return np.einsum('ea,nab,hb->neh',u,core.reshape(len(x),u.shape[1],v.shape[1]),v)+params[key+'__mean']
    return (core@params[key+'__basis'].T).reshape(x.shape)+params[key+'__mean']


def predict(params,key,features,own):
    z=(features-params[key+'__scale_mean'])/params[key+'__scale_scale']
    z=np.concatenate([z,np.eye(4)[own]],axis=1);classes=params[key+'__classes']
    if len(classes)==1:return np.full(len(z),classes[0],int)
    values=z@params[key+'__coef'].T+params[key+'__intercept']
    indices=(values[:,0]>0).astype(int) if len(classes)==2 else values.argmax(-1)
    return classes[indices]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--previous',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    assert json.loads((source/'status.json').read_text())['state']=='passed'
    tasks=['sst2','mrpc','rte','cola'];seeds=[42,43,44];summaries=[];rows=[];checks=[]
    for seed in seeds:
        folder=source/f'seed{seed}';s=json.loads((folder/'summary.json').read_text());summaries.append(s)
        params=np.load(folder/'fitted-parameters.npz');cache=np.load(folder/'response-cache.npz');pred=np.load(folder/'predictions.npz')
        ids=json.loads((folder/'split-ids.json').read_text());old_ids=json.loads((Path(a.previous)/f'seed{seed}/split-ids.json').read_text())
        assert ids==old_ids
        for t in tasks:
            parts=[set(ids[role][t]) for role in ['coefficient_fit','coefficient_selection','validation']]
            assert not(parts[0]&parts[1] or parts[0]&parts[2] or parts[1]&parts[2])
        assert s['counts']=={'coefficient_fit':512,'coefficient_selection':512,'validation':1024}
        for key,record in s['representations'].items():
            for role,field in [('coefficient_fit','fit_error'),('coefficient_selection','selection_error'),('validation','final_error')]:
                x=cache[role];mean=params[key+'__mean'];value=float(np.square(x-reconstruct(params,key,x)).sum()/np.square(x-mean).sum())
                assert np.isclose(value,record[field],atol=1e-10,rtol=1e-8),(seed,key,role)
            if key.startswith('csa'):
                assert np.min(np.diff(record['history']))>=-1e-7*max(record['history'])
                for k in ['factor0','factor1']:
                    f=params[key+'__'+k];np.testing.assert_allclose(f.T@f,np.eye(f.shape[1]),atol=1e-9)
            rows.append({'seed':seed,'method':key,'kind':record['kind'],'latent':record['storage']['latent'],
                         'scalars':record['storage']['total'],'ratio':record['storage']['raw']/record['storage']['total'],
                         'fit_error':record['fit_error'],'final_error':record['final_error'],'native':s['routing'][key]['score']})
        oldsel=np.load(Path(a.previous)/f'seed{seed}/coefficient_selection-cache.npz')
        for key,locked in s['selection'].items():
            best=max(locked['candidates'],key=lambda c:(c['score'],-c['C']));assert best==locked['selected']
            for role,logits in [('coefficient_selection',oldsel),('validation',pred)]:
                own=np.concatenate([np.full(len(logits[t+'__labels']),i,int) for i,t in enumerate(tasks)])
                features=np.concatenate([logits[t+'__base'] for t in tasks]) if key=='base' and role!='validation' else None
                if key=='base' and role=='validation':
                    oldfinal=np.load(Path(a.previous)/f'seed{seed}/validation-cache.npz');features=np.concatenate([oldfinal[t+'__base'] for t in tasks])
                if key!='base':features=project(params,key,cache[role])
                routes=predict(params,key,features,own);score,_=native(logits,routes,tasks)
                if role=='coefficient_selection':assert abs(score-best['score'])<1e-10
                else:np.testing.assert_array_equal(routes,pred[key+'__routes'])
        for key,value in s['routing'].items():
            score,per_task=native(pred,pred[key+'__routes'],tasks);assert abs(score-value['score'])<1e-10
            for t in tasks:
                assert abs(per_task[t]['accuracy']-value['tasks'][t]['metrics']['accuracy'])<1e-10
        checks.append({'seed':seed,'state':'passed','projection_reconstruction_reload':True,'router_reload_and_selection_score':True,'ids_match_previous_roles':True})
        for name in ['summary.json','selection-lock.json','split-ids.json','predictions.npz','status.json']:
            (out/f'seed{seed}-{name}').write_bytes((folder/name).read_bytes())
    means={}
    for key in summaries[0]['representations']:
        records=[s['representations'][key] for s in summaries]
        means[key]={'storage':records[0]['storage'],'final_error_mean':float(np.mean([v['final_error'] for v in records])),
                    'final_error_sd':float(np.std([v['final_error'] for v in records],ddof=1)),
                    'fit_error_mean':float(np.mean([v['fit_error'] for v in records]))}
    routes={key:{'mean':float(np.mean([s['routing'][key]['score'] for s in summaries])),
                 'sd':float(np.std([s['routing'][key]['score'] for s in summaries],ddof=1))} for key in summaries[0]['routing']}
    result={'state':'passed','seeds':seeds,'shape':summaries[0]['shape'],'counts':summaries[0]['counts'],'representations':means,'routing':routes,'audit':checks}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    with (out/'per-seed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name in ['plan.json','source-sha256.json','status.json']:(out/name).write_bytes((source/name).read_bytes())
    for name in ['code/csa_bridge.py','code/test_csa_bridge.py','code/summarize_csa_bridge.py','part1-reproduction-20261025/code/csa_tensor.py']:
        (out/('source-'+Path(name).name)).write_bytes((ROOT/name).read_bytes())
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11})
    fig,ax=plt.subplots(1,2,figsize=(13,4.7))
    for kind,color,marker in [('pca','#ba7887','s'),('csa','#208a78','o')]:
        items=[(k,v) for k,v in means.items() if k.startswith(kind)]
        items.sort(key=lambda kv:kv[1]['storage']['total']);x=[v['storage']['total']*8/1024 for _,v in items]
        ax[0].errorbar(x,[v['final_error_mean'] for _,v in items],yerr=[v['final_error_sd'] for _,v in items],fmt=marker+'-',color=color,label=kind.upper(),capsize=3)
        ax[1].errorbar(x,[routes[k]['mean'] for k,_ in items],yerr=[routes[k]['sd'] for k,_ in items],fmt=marker+'-',color=color,label=kind.upper(),capsize=3)
        if kind=='csa':
            for (k,v),xx in zip(items,x):ax[0].annotate(k[3:],(xx,v['final_error_mean']),xytext=(4,5),textcoords='offset points',fontsize=8)
    for a in ax:a.set_xscale('log');a.set_xlabel('Mean + basis + 512 codes (KiB, float64)');a.grid(alpha=.2);a.legend()
    ax[0].set_ylabel('Relative squared reconstruction error');ax[0].set_title('Internal-final response reconstruction')
    ax[1].set_ylabel('Native task score');ax[1].set_title('Offline expert routing from compressed responses')
    ax[1].axhline(routes['task_id']['mean'],color='black',ls='--',label='Task ID')
    ax[1].axhline(routes['raw_delta']['mean'],color='#5b7184',ls=':',label='Raw delta');ax[1].legend(fontsize=9)
    fig.text(.5,.015,'Mean +/- sample SD, three fixed teacher seeds. All-expert features; no serving-cost claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'csa-storage-quality.png',dpi=180);plt.close(fig)
    keys=['csa2x8','pca16','pca'+str(summaries[0]['representations']['csa2x8']['budget_pca'][3:]),'raw_delta','base','task_id']
    fig,ax=plt.subplots(figsize=(10,4.5));positions=np.arange(len(keys))
    ax.bar(positions,[routes[k]['mean'] for k in keys],yerr=[routes[k]['sd'] for k in keys],capsize=4,color=['#208a78','#ba7887','#c89aa4','#5b7184','#8b9298','#333333'])
    ax.set_xticks(positions,keys);ax.set_ylim(0,1);ax.set_ylabel('Native task score');ax.set_title('A fixed illustration: CSA 2 x 8 and its PCA controls')
    fig.text(.5,.015,'All ranks are reported. The illustration is not a final-selected winner. GT oracle is excluded.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'csa-routing-controls.png',dpi=180);plt.close(fig)
    tex=['\\begin{table}[htbp]\\centering\\small','\\caption{Direct CSA transfer to expert-by-feature response matrices. Storage includes a fit mean, bases and 512 codes in float64. Reconstruction errors are relative squared errors on the previously explored internal final. Routing requires all four experts. Mean $\\pm$ sample SD over three fixed teacher seeds.}','\\begin{tabular}{lrrrr}\\toprule','Representation & Latent & KiB & Rel. error & Native score \\\\ \\midrule']
    for key in keys:
        v=means.get(key);score=routes[key];name=key.replace('_','\\_')
        if v:
            tex.append(f"{name} & {v['storage']['latent']} & {v['storage']['total']*8/1024:.1f} & {v['final_error_mean']:.4f} & {score['mean']:.4f} $\\pm$ {score['sd']:.4f} \\\\")
        else:tex.append(f"{name} & -- & -- & -- & {score['mean']:.4f} $\\pm$ {score['sd']:.4f} \\\\")
    tex+=['\\bottomrule\\end{tabular}\\end{table}']
    (out/'table-csa-bridge.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
