"""Independently audit saved predictions and publish the robustness follow-up."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics as st
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--sensitivity',required=True)
    p.add_argument('--controls',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    previous,sensitivity,controls,out=map(Path,[args.previous,args.sensitivity,args.controls,args.output])
    out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];report=root/'report-20261109'
    report.mkdir(exist_ok=True)
    read=lambda p:json.loads(p.read_text());seeds=[42,43,44];tasks=['sst2','mrpc','rte','cola']
    assert read(sensitivity/'status.json')['state']==read(controls/'status.json')['state']=='passed'
    rows=[];per_task=[];grid=[];selections=[];official_ids=None
    def audit(logits,labels,stored,task):
        guessed=logits.argmax(-1)
        m={'accuracy':float(accuracy_score(labels,guessed)),'f1':float(f1_score(labels,guessed,zero_division=0)),
           'mcc':float(matthews_corrcoef(labels,guessed))}
        for k,v in m.items():assert abs(stored['metrics'][k]-v)<1e-12,(task,k)
        primary=m['mcc'] if task=='cola' else (m['accuracy']+m['f1'])/2 if task=='mrpc' else m['accuracy']
        if 'primary_score' in stored:assert abs(stored['primary_score']-primary)<1e-12
        return {'primary':primary,**m}
    for seed in seeds:
        current=read(sensitivity/f'seed{seed}/summary.json');old=read(previous/f'seed{seed}/summary.json')
        result=read(controls/f'seed{seed}/summary.json');lock=read(controls/f'seed{seed}/selection-lock.json')
        assert current['state']==old['state']==result['state']=='passed' and lock==result['selection']
        assert lock['locked_before_reading_cached_final_and_loading_official']
        selections.append({'seed':seed,'hidden_lambda':lock['static']['hidden']['lambda'],'shuffled_lambda':lock['static']['shuffled']['lambda'],
                           'confidence_threshold':lock['confidence_selected']['threshold'],'benefit_alpha':lock['benefit_selected']['alpha'],
                           'benefit_threshold':lock['benefit_selected']['threshold']})
        ids=read(controls/f'seed{seed}/official-data-manifest.json')
        if official_ids is None:official_ids=ids
        else:assert ids==official_ids,'Data differ across seeds'
        cp=np.load(sensitivity/f'seed{seed}/predictions.npz',allow_pickle=False)
        op=np.load(previous/f'seed{seed}/predictions.npz',allow_pickle=False)
        # Audit every fixed-lambda profile, including the reused 0 and 1 endpoints.
        for family,candidates in lock['lambda_candidates'].items():
            best=sorted(candidates,key=lambda x:(-x['selection_score'],x['lambda']))[0]
            assert best==lock['static'][family]
            for c in candidates:
                source=old if c['source']=='previous' else current;pred=op if c['source']=='previous' else cp
                v=source['final_static'][c['profile']];scores=[];acc=[]
                assert c['selection_score']==source['fits'][c['profile']]['selected']['score']
                for task in tasks:
                    m=audit(pred[c['profile']+'__'+task],pred['labels__'+task],v['tasks'][task],task);scores.append(m['primary']);acc.append(m['accuracy'])
                grid.append({'seed':seed,'family':family,'lambda':c['lambda'],'selection_native':c['selection_score'],
                             'final_native':st.mean(scores),'final_accuracy':st.mean(acc),'final_worst':min(acc),'selected':c==best,'step':c['step']})
        cp.close();op.close()
        with np.load(controls/f'seed{seed}/predictions.npz',allow_pickle=False) as pred:
            for role,categories in result['results'].items():
                for category,methods in categories.items():
                    for method,value in methods.items():
                        scores=[];acc=[];helped=[];harmed=[];offset=0
                        for task in tasks:
                            labels=pred[role+'__labels__'+task];n=len(labels)
                            if category=='static':logits=pred[role+'__'+method+'__'+task]
                            else:
                                route=pred[role+'__route__'+method][offset:offset+n]
                                experts=pred[role+'__expert_logits__'+task]
                                logits=experts[route,np.arange(n)]
                                own=experts[tasks.index(task)].argmax(-1)==labels;correct=logits.argmax(-1)==labels
                                h=float((correct & ~own).mean());d=float((~correct & own).mean());helped.append(h);harmed.append(d)
                                assert abs(h-value['impact_vs_own'][task]['helps_fraction'])<1e-12
                                assert abs(d-value['impact_vs_own'][task]['harms_fraction'])<1e-12
                                if method=='oracle':assert np.all(correct | ~(experts.argmax(-1)==labels).any(0))
                            m=audit(logits,labels,value['tasks'][task],task);scores.append(m['primary']);acc.append(m['accuracy'])
                            per_task.append({'seed':seed,'role':role,'category':category,'method':method,'task':task,'samples':n,**m});offset+=n
                        row={'seed':seed,'role':role,'category':category,'method':method,'macro_primary':st.mean(scores),
                             'mean_accuracy':st.mean(acc),'worst_accuracy':min(acc),'switch_fraction':value.get('switch_fraction',0.),
                             'helps_macro':st.mean(helped) if helped else 0.,'harms_macro':st.mean(harmed) if harmed else 0.}
                        assert abs(value['score']-row['macro_primary'])<1e-12
                        assert abs(value['mean_accuracy']-row['mean_accuracy'])<1e-12
                        if category=='routing':
                            assignment=pred[role+'__route__'+method]
                            own=np.concatenate([np.full(len(pred[role+'__labels__'+t]),i) for i,t in enumerate(tasks)])
                            assert abs(float((assignment!=own).mean())-row['switch_fraction'])<1e-12
                        rows.append(row)
    aggregate={}
    for role in ['internal','official']:
        aggregate[role]={}
        for category in ['static','routing']:
            aggregate[role][category]={}
            for method in sorted({r['method'] for r in rows if r['role']==role and r['category']==category}):
                values=[r for r in rows if (r['role'],r['category'],r['method'])==(role,category,method)]
                aggregate[role][category][method]={k:{'mean':st.mean(r[k] for r in values),'sample_sd':st.stdev(r[k] for r in values)}
                    for k in ['macro_primary','mean_accuracy','worst_accuracy','switch_fraction','helps_macro','harms_macro']}
    paired=[]
    for role in ['internal','official']:
        for seed in seeds:
            by={(r['category'],r['method']):r for r in rows if r['role']==role and r['seed']==seed}
            paired.append({'seed':seed,'role':role,
                'hidden_minus_ce':by['static','selected_hidden']['macro_primary']-by['static','ce_fp32']['macro_primary'],
                'shuffled_minus_ce':by['static','selected_shuffled']['macro_primary']-by['static','ce_fp32']['macro_primary'],
                'hidden_minus_shuffled':by['static','selected_hidden']['macro_primary']-by['static','selected_shuffled']['macro_primary'],
                'confidence_minus_taskid_accuracy':by['routing','confidence_gate']['mean_accuracy']-by['routing','task_id']['mean_accuracy'],
                'benefit_minus_taskid_accuracy':by['routing','benefit_gate']['mean_accuracy']-by['routing','task_id']['mean_accuracy']})
    summary={'state':'passed','aggregate':aggregate,'per_seed':rows,'paired':paired,'selected':selections,'official_data':official_ids,
             'audit':'All saved final logits/labels/routes, native metrics, helps/harms, switch fractions, oracle binary ceiling, lambda choices and cross-seed IDs checked.',
             'scope':'Exploratory after internal and historical official validation inspection. Three teacher seeds on one fixed partition; recovery/sensitivity fits do not increase n.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name,values in [('per-seed.csv',rows),('per-task.csv',per_task),('lambda-grid.csv',grid),('paired.csv',paired),('selected.csv',selections)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.6))
    for ax,key,title in zip(axes,['selection_native','final_native'],['Selection role','Previously explored internal final']):
        for family,color,label in [('hidden','#248a74','Aligned'),('shuffled','#aa4f62','Shuffled')]:
            vals=[[r[key] for r in grid if r['family']==family and r['lambda']==w] for w in [0.,.01,.1,1.]]
            ax.errorbar(range(4),[st.mean(v) for v in vals],yerr=[st.stdev(v) for v in vals],marker='o',capsize=3,color=color,label=label)
        ax.set_xticks(range(4),['0 (CE)','0.01','0.1','1']);ax.set_ylim(.5,.85);ax.set_xlabel('Hidden-loss weight');ax.set_ylabel('Native task score');ax.set_title(title);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Loss-weight sensitivity: same FP32 coefficients and matched shuffle grid')
    fig.text(.5,.01,'Mean +/- sample SD over three fixed teacher seeds; selection across lambda uses selection only. Exploratory.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(out/'lambda-sensitivity.png',dpi=180);plt.close(fig)
    def chart(specs,labels,filename,title):
        fig,axes=plt.subplots(1,2,figsize=(12.2,4.8))
        colors=['#526d82','#8899a6','#8899a6','#8899a6','#8899a6','#d98a34','#248a74','#aa4f62'][:len(specs)]
        if specs[-1]==('routing','oracle'):colors=['#526d82','#8899a6','#d98a34','#248a74','#aa4f62','#8a8a8a']
        for ax,key,label in zip(axes,['macro_primary','worst_accuracy'],['Native task score','Worst-task accuracy']):
            vals=[aggregate['official'][cat][m][key] for cat,m in specs]
            ax.bar(range(len(specs)),[v['mean'] for v in vals],yerr=[v['sample_sd'] for v in vals],capsize=3,color=colors)
            ax.set_xticks(range(len(specs)),labels,rotation=25,ha='right');ax.set_ylim(0,1);ax.set_title(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle(title);fig.text(.5,.01,'Cleaned official GLUE validation; previously explored tasks. Mean +/- sample SD; three teacher seeds.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(out/filename,dpi=180);plt.close(fig)
    specs=[('routing','task_id')]+[('static',m) for m in ['average','task_arithmetic','ties','dare','ce_fp32','selected_hidden','selected_shuffled']]
    chart(specs,['Individual','Average','TA','TIES','DARE+TA','CE FP32','Selected H','Selected shuffle'],'official-static.png','Locked static methods on the cleaned official validation split')
    methods=['task_id','source_router','expert_router','confidence_gate','benefit_gate','oracle']
    chart([('routing',m) for m in methods],['Task ID','Source','Expert','Confidence gate','Benefit gate','Oracle (GT)'],'official-routing.png','Conservative routing: selection constraints do not guarantee final safety')
    fig,ax=plt.subplots(figsize=(8,4.6))
    for i,m in enumerate(methods[1:]):
        v=aggregate['official']['routing'][m];h=v['helps_macro']['mean']*100;d=v['harms_macro']['mean']*100
        ax.bar(i,h,color='#248a74');ax.bar(i,-d,color='#aa4f62',alpha=.7)
        ax.text(i,h+.3,f'+{h:.2f}',ha='center',fontsize=9);ax.text(i,-d-.4,f'-{d:.2f}',ha='center',va='top',fontsize=9)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(5),['Source','Expert','Confidence','Benefit','Oracle (GT)']);ax.set_ylabel('Macro fraction helped / harmed vs own (%)');ax.set_title('Actual switching consequences on cleaned official validation');ax.margins(y=.25);fig.tight_layout();fig.savefig(out/'official-routing-impact.png',dpi=180);plt.close(fig)
    def fmt(v):return f"{v['mean']:.4f} $\\pm$ {v['sample_sd']:.4f}"
    lines=['\\begin{table}[H]\\centering\\small','\\caption{Fixed lambda sensitivity. Same precision/capacity; three teacher seeds. Zero is CE. Lambda grid increases selection budget.}',
           '\\begin{tabular}{lrrrr}\\toprule','$\\lambda$ & Aligned select & Shuffle select & Aligned final & Shuffle final\\\\\\midrule']
    for w in [0.,.01,.1,1.]:
        values=[]
        for family,key in [('hidden','selection_native'),('shuffled','selection_native'),('hidden','final_native'),('shuffled','final_native')]:
            a=[r[key] for r in grid if r['family']==family and r['lambda']==w];values.append(fmt({'mean':st.mean(a),'sample_sd':st.stdev(a)}))
        lines.append(str(w)+' & '+' & '.join(values)+'\\\\')
    lines+=['\\bottomrule\\end{tabular}\\end{table}'];(report/'table-sensitivity-v2.tex').write_text('\n'.join(lines)+'\n')
    labels=['Individual / task ID','Average','Task Arithmetic','TIES','DARE+TA','CE FP32','Selected aligned H','Selected shuffled H','Source router','Expert router','Confidence gate','Benefit gate','Oracle (final GT)']
    official_specs=specs+[('routing',m) for m in methods[1:]]
    lines=['\\begin{table}[H]\\centering\\small','\\caption{Locked methods on cleaned official GLUE validation. Prior historical exploration remains disclosed; oracle uses final GT.}',
           '\\begin{tabular}{lrrr}\\toprule','Method & Native score & Mean accuracy & Worst accuracy\\\\\\midrule']
    for label,(cat,m) in zip(labels,official_specs):lines.append(label+' & '+' & '.join(fmt(aggregate['official'][cat][m][k]) for k in ['macro_primary','mean_accuracy','worst_accuracy'])+'\\\\')
    lines+=['\\bottomrule\\end{tabular}\\end{table}'];(report/'table-official-v2.tex').write_text('\n'.join(lines)+'\n')
    def effect(role,key):return st.mean(r[key] for r in paired if r['role']==role)
    counts={t:official_ids['tasks'][t]['retained_count'] for t in tasks}
    zero=sum(r['hidden_lambda']==0 for r in selections)
    macros={'SensitivityVtwo':f'Selection chooses zero hidden weight in {zero}/3 seeds. The selected aligned-grid minus CE native-score difference is {effect("internal","hidden_minus_ce"):+.6f} on internal final and {effect("official","hidden_minus_ce"):+.6f} on cleaned official validation. These are pipeline comparisons with additional selection candidates, not isolated evidence of a unique functional effect.',
            'OfficialVtwo':f'The retained validation counts are {counts["sst2"]}, {counts["mrpc"]}, {counts["rte"]}, and {counts["cola"]} for SST-2, MRPC, RTE, and CoLA. The split was previously explored historically; full-tuple overlap filtering does not remove all near duplicates or pretraining contamination.',
            'GatesVtwo':f'On cleaned official validation, confidence and benefit gates change macro mean accuracy versus task ID by {100*effect("official","confidence_minus_taskid_accuracy"):+.3f} and {100*effect("official","benefit_minus_taskid_accuracy"):+.3f} percentage points. A selection constraint is not a guarantee on final inputs; the actual help/harm rates and fallback choices determine the interpretation.',
            'AbstractVtwo':f'A further fixed-grid sensitivity and conservative-routing study evaluates a larger, exact-tuple-cleaned official-validation split. Selected hidden versus CE changes native score by {effect("official","hidden_minus_ce"):+.4f}; the benefit gate changes macro accuracy versus task ID by {100*effect("official","benefit_minus_taskid_accuracy"):+.2f} points, with exploratory scope.',
            'ConclusionVtwo':f'The larger cleaned-validation follow-up gives selected hidden-minus-CE native change {effect("official","hidden_minus_ce"):+.4f} and benefit-gate-minus-task-ID accuracy change {100*effect("official","benefit_minus_taskid_accuracy"):+.2f} points. These results describe a selected pipeline under known-task access, not a new blind benchmark or equal-budget mechanism proof.'}
    (report/'robustness-results.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    for seed in seeds:
        for directory,prefix,names in [(sensitivity,'sensitivity',['summary.json','selection.json','split-ids.json','fit-control.json','predictions.npz']),
                                       (controls,'controls',['summary.json','selection-lock.json','official-data-manifest.json','predictions.npz'])]:
            for name in names:shutil.copyfile(directory/f'seed{seed}'/name,out/f'{prefix}-s{seed}-{name}')
    for directory,prefix in [(sensitivity,'sensitivity'),(controls,'controls')]:
        for name in ['plan.json','status.json','source-sha256.json']:shutil.copyfile(directory/name,out/(prefix+'-'+name))
    for name in ['activation_extension.py','robustness_controls.py','test_activation_extension.py','test_robustness_controls.py','summarize_robustness.py']:
        shutil.copyfile(root/'code'/name,out/('source-'+name))
    (out/'README.md').write_text('# Sensitivity, conservative routing and cleaned official-validation evidence\n\nExploratory follow-up after earlier final inspection. Three fixed teacher seeds; additional fits do not increase n.\n\n- summary/CSVs: independently audited predictions, paired effects, lambda/threshold choices and actual retained data counts.\n- lambda-sensitivity/official-static/official-routing/official-routing-impact PNG: presentation figures.\n- sensitivity-s*/controls-s*: complete histories, lock files, IDs and derived logits/labels/routes. Raw text, feature caches and weights stay outside Git.\n- source/plan/preflight records: exact execution provenance.\n- manifest: exact SHA256 for all other files.\n\nLambda selection has more candidates than CE; aligned and shuffled grids match. Gate constraints apply to selection only. Oracle uses final labels and is an accuracy ceiling only for this binary protocol. Official validation was explored historically and is not a pristine blind test or new OOD claim.\n')
    for source in ['sensitivity-preflight.json','robustness-preflight.json','robustness-final-validation.json']:
        if (root/'outputs'/source).exists():shutil.copyfile(root/'outputs'/source,out/source)
    (out/'manifest.json').write_text(json.dumps({'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}},indent=2))
    print(json.dumps({'state':'passed','official_counts':counts,'selected':selections,'paired':paired}))


if __name__=='__main__':main()
