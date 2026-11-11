"""Reload every discriminant, re-audit selection and count actual switch outcomes."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from summarize_csa_bridge import project

ROOT=Path(__file__).resolve().parents[1];TASKS=['sst2','mrpc','rte','cola']


def audit_metrics(cache,routes):
    from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef
    native=[];acc=[];helped=[];harmed=[];own=[];offset=0
    for i,t in enumerate(TASKS):
        y=cache[t+'__labels'];n=len(y);logits=cache[t+'__logits'];allpred=logits.argmax(-1)
        pred=allpred[routes[offset:offset+n],np.arange(n)];correct=pred==y;original=allpred[i]==y
        a=accuracy_score(y,pred);native.append(matthews_corrcoef(y,pred) if t=='cola' else (a+f1_score(y,pred,zero_division=0))/2 if t=='mrpc' else a)
        acc.append(a);helped.append(np.mean(correct&~original));harmed.append(np.mean(~correct&original));own.extend([i]*n);offset+=n
    return {'native':float(np.mean(native)),'accuracy':float(np.mean(acc)),'worst_accuracy':float(min(acc)),
            'helps':float(np.mean(helped)),'harms':float(np.mean(harmed)),'switch_fraction':float(np.mean(routes!=np.array(own)))}


def get_prob(state,key,alpha,target,x):
    prefix=key+'__'+str(alpha)+'__'+target
    classes=state[prefix+'__classes'];logits=x@state[prefix+'__coef'].T+state[prefix+'__intercept']
    logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
    return p,classes


def routes_for(state,key,family,candidate,x,own):
    if candidate.get('always_own'):return own.copy()
    alpha=candidate['alpha'];threshold=candidate['threshold']
    if family in ['direct_lda','task_prior']:
        p,classes=get_prob(state,key,alpha,'expert',x);prob=np.zeros((len(x),4));prob[:,classes.astype(int)]=p
        if family=='direct_lda':return prob.argmax(1)
        gains=prob-prob[np.arange(len(own)),own,None]
    else:gains=np.column_stack([get_prob(state,key,alpha,'benefit'+str(j),x)[0]@get_prob(state,key,alpha,'benefit'+str(j),x)[1] for j in range(4)])
    gains[np.arange(len(own)),own]=-np.inf;alt=gains.argmax(1)
    return np.where(gains[np.arange(len(own)),alt]>threshold,alt,own)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--bridge',required=True);p.add_argument('--previous',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.input);bridge=Path(a.bridge);previous=Path(a.previous);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    assert json.loads((source/'status.json').read_text())['state']=='passed'
    plan=json.loads((source/'plan.json').read_text());rows=[];audits=[];all_methods={}
    for seed in plan['seeds']:
        folder=source/f'seed{seed}';state=np.load(folder/'fitted-discriminants.npz');projection=np.load(bridge/f'seed{seed}/fitted-parameters.npz');responses=np.load(bridge/f'seed{seed}/response-cache.npz')
        locked=json.loads((folder/'selection-lock.json').read_text());summary=json.loads((folder/'summary.json').read_text());predictions=np.load(folder/'predictions.npz')
        assert json.loads((folder/'split-ids.json').read_text())==json.loads((bridge/f'seed{seed}/split-ids.json').read_text())
        selection=np.load(previous/f'seed{seed}/coefficient_selection-cache.npz');final=np.load(previous/f'seed{seed}/validation-cache.npz')
        def own(cache):return np.concatenate([np.full(len(cache[t+'__labels']),i,int) for i,t in enumerate(TASKS)])
        ownsel,ownfinal=own(selection),own(final);reference=audit_metrics(selection,ownsel);basefinal=audit_metrics(final,ownfinal)
        candidates_checked=0
        for key,families in locked.items():
            features={}
            for role,cache,ident in [('coefficient_selection',selection,ownsel),('validation',final,ownfinal)]:
                raw=np.concatenate([cache[t+'__base'] for t in TASKS]) if key=='base' else project(projection,key,responses[role])
                features[role]=np.column_stack([(raw-state[key+'__mean'])/state[key+'__scale'],np.eye(4)[ident]])
            for family,record in families.items():
                eligible=[]
                for i,candidate in enumerate(record['candidates']):
                    routes=routes_for(state,key,family,candidate,features['coefficient_selection'],ownsel);value=audit_metrics(selection,routes)
                    for metric,n in value.items():assert abs(n-candidate['metrics'][metric])<1e-10,(seed,key,family,metric)
                    safe=family!='direct_lda';ok=not safe or (value['native']>=reference['native']-1e-12 and value['accuracy']>=reference['accuracy']-1e-12 and value['worst_accuracy']>=reference['worst_accuracy']-plan['worst_accuracy_tolerance']-1e-12)
                    assert ok==candidate['eligible']
                    if ok:eligible.append(i)
                    candidates_checked+=1
                best=max(eligible,key=lambda i:(record['candidates'][i]['metrics']['native'],record['candidates'][i]['metrics']['accuracy'] if safe else 0.,-record['candidates'][i]['metrics']['switch_fraction'] if safe else 0.,record['candidates'][i]['alpha'],record['candidates'][i].get('threshold',-1) or 0.))
                assert best==record['selected'];chosen=record['candidates'][best]
                routes=routes_for(state,key,family,chosen,features['validation'],ownfinal)
                np.testing.assert_array_equal(routes,predictions[key+'__'+family+'__routes']);metrics=audit_metrics(final,routes)
                for name,value in metrics.items():assert abs(value-summary['methods'][key][family][name])<1e-10
                assert abs(metrics['helps']-metrics['harms']-(metrics['accuracy']-basefinal['accuracy']))<1e-12
                rows.append({'seed':seed,'representation':key,'family':family,'alpha':chosen['alpha'],'threshold':chosen['threshold'],
                             'always_own':bool(chosen.get('always_own',False)),**metrics,'accuracy_delta':metrics['accuracy']-basefinal['accuracy'],'native_delta':metrics['native']-basefinal['native']})
            old=np.load(bridge/f'seed{seed}/predictions.npz');metrics=audit_metrics(final,old[key+'__routes'])
            rows.append({'seed':seed,'representation':key,'family':'previous_logistic','alpha':None,'threshold':None,'always_own':False,**metrics,'accuracy_delta':metrics['accuracy']-basefinal['accuracy'],'native_delta':metrics['native']-basefinal['native']})
        rows.append({'seed':seed,'representation':'task_id','family':'reference','alpha':None,'threshold':None,'always_own':True,**basefinal,'accuracy_delta':0.,'native_delta':0.})
        for t in TASKS:
            np.testing.assert_array_equal(predictions[t+'__labels'],final[t+'__labels']);np.testing.assert_allclose(predictions[t+'__logits'],final[t+'__logits'],atol=1e-10)
        audits.append({'seed':seed,'state':'passed','selection_candidates_recomputed':candidates_checked,'all_final_routes_reload_equal':True,'help_minus_harm_identity':True})
        for name in ['summary.json','selection-lock.json','split-ids.json','predictions.npz','status.json']:(out/f'seed{seed}-{name}').write_bytes((folder/name).read_bytes())
    for key,family in sorted(set((r['representation'],r['family']) for r in rows)):
        subset=[r for r in rows if r['representation']==key and r['family']==family];assert len(subset)==3
        aggregate={metric:{'mean':float(np.mean([r[metric] for r in subset])),'sd':float(np.std([r[metric] for r in subset],ddof=1))} for metric in ['native','accuracy','worst_accuracy','switch_fraction','helps','harms','accuracy_delta','native_delta']}
        aggregate.update(always_own_seeds=sum(r['always_own'] for r in subset),nonzero_switch_seeds=sum(r['switch_fraction']>0 for r in subset))
        all_methods.setdefault(key,{})[family]=aggregate
    result={'state':'passed','methods':all_methods,'audits':audits,'seeds':plan['seeds'],'role':'Previously explored internal final; 1024 balanced examples per seed.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    with (out/'per-seed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name in ['plan.json','source-sha256.json','status.json']:(out/name).write_bytes((source/name).read_bytes())
    for name in ['csa_discriminant.py','test_csa_discriminant.py','summarize_csa_discriminant.py']:(out/('source-'+name)).write_bytes((ROOT/'code'/name).read_bytes())
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=['csa2x8','pca16','pca3','raw_delta','base'];families=['previous_logistic','direct_lda','task_prior','benefit_lda'];colors=['#8c9398','#bd8a95','#5b7184','#208a78']
    fig,ax=plt.subplots(figsize=(11,4.8));x=np.arange(len(keys));width=.19
    for i,(family,color) in enumerate(zip(families,colors)):
        records=[all_methods[key][family]['native'] for key in keys]
        ax.bar(x+(i-1.5)*width,[v['mean'] for v in records],width,yerr=[v['sd'] for v in records],capsize=2,label=family,color=color)
    ax.axhline(all_methods['task_id']['reference']['native']['mean'],color='black',ls='--',label='Task ID')
    ax.set_xticks(x,keys);ax.set_ylim(0,1);ax.set_ylabel('Native task score');ax.legend(ncol=3,fontsize=9)
    ax.set_title('Discriminative CSA/PCA routing: frozen selection, matched method grids')
    fig.text(.5,.01,'Three fixed teacher seeds; previously explored internal final; all-expert offline responses.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(out/'discriminant-quality.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,family in zip(axes,['task_prior','benefit_lda']):
        helped=[all_methods[k][family]['helps']['mean']*100 for k in keys];harmed=[-all_methods[k][family]['harms']['mean']*100 for k in keys]
        ax.bar(x,helped,color='#208a78',label='Helped');ax.bar(x,harmed,color='#bd8a95',label='Harmed');ax.axhline(0,color='black',lw=.8)
        for i,k in enumerate(keys):ax.annotate(f"switch {all_methods[k][family]['switch_fraction']['mean']*100:.2f}%",(i,helped[i]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(x,keys,rotation=20);ax.set_ylabel('Task-macro examples (%)');ax.set_title(family);ax.legend(fontsize=9);ax.margins(y=.3)
    fig.text(.5,.01,'Zero gain can mean complete fallback. Selection floors do not guarantee final safety.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'discriminant-switch-impact.png',dpi=180);plt.close(fig)
    examples=[('task_id','reference'),('csa2x8','previous_logistic'),('csa2x8','direct_lda'),('csa2x8','task_prior'),('csa2x8','benefit_lda'),('pca16','benefit_lda'),('pca3','benefit_lda'),('raw_delta','benefit_lda')]
    tex=['\\begin{table}[htbp]\\centering\\scriptsize','\\caption{Discriminative follow-up on the same internal final, three fixed teacher seeds. All candidates use matched grids across representations. Switch/help/harm are mean percentages; native score is mean $\\pm$ sample SD. Own is the number of seeds selecting always-own.}','\\begin{tabular}{llrrrrr}\\toprule','Features & Method & Native & Switch & Help & Harm & Own \\\\ \\midrule']
    for key,family in examples:
        v=all_methods[key][family];score=v['native'];label=family.replace('_','\\_');key=key.replace('_','\\_')
        tex.append(f"{key} & {label} & {score['mean']:.4f} $\\pm$ {score['sd']:.4f} & {v['switch_fraction']['mean']*100:.2f} & {v['helps']['mean']*100:.2f} & {v['harms']['mean']*100:.2f} & {v['always_own_seeds']}/3 \\\\")
    tex+=['\\bottomrule\\end{tabular}\\end{table}'];(out/'table-csa-discriminant.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps({'state':'passed','audits':audits,'illustration':all_methods['csa2x8'],'task_id':all_methods['task_id']},indent=2))


if __name__=='__main__':main()
