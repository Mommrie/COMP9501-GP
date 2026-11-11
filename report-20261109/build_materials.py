"""Audit completed extension predictions, build tables/figures, and fill report text."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    import numpy as np
    from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--extension',required=True)
    p.add_argument('--artifacts',required=True);args=p.parse_args()
    study,ext,out=Path(args.study),Path(args.extension),Path(args.artifacts);out.mkdir(parents=True,exist_ok=False)
    read=lambda path:json.loads(path.read_text())
    assert read(ext/'status.json')['state']=='passed'
    cfg=read(study/'locked-config.json');tasks=cfg['tasks'];seeds=[42,43,44]
    rows=[];fitrows=[];audit=[];routes=[]
    for seed in seeds:
        folder=ext/f'seed{seed}';s=read(folder/'summary.json');assert s['state']=='passed'
        assert read(folder/'selection.json')['locked_before_final_load']
        assert s['partition_sha256']==cfg['partition_manifest_sha256']
        pred=np.load(folder/'predictions.npz',allow_pickle=False)
        for category,key in [('static','final_static'),('router','final_router')]:
            for method,value in s[key].items():
                per=[]
                for t in tasks:
                    labels=pred['labels__'+t]
                    if category=='static':logits=pred[method+'__'+t]
                    else:
                        offset=tasks.index(t)*len(labels);assignment=pred['route__'+method][offset:offset+len(labels)]
                        logits=pred['expert_logits__'+t][assignment,np.arange(len(labels))]
                    guessed=logits.argmax(-1)
                    metrics={'accuracy':float(accuracy_score(labels,guessed)),
                             'f1':float(f1_score(labels,guessed,zero_division=0)),
                             'mcc':float(matthews_corrcoef(labels,guessed))}
                    for metric,v in metrics.items():assert abs(value['tasks'][t]['metrics'][metric]-v)<1e-12,(seed,method,t,metric)
                    primary=metrics['mcc'] if t=='cola' else (metrics['accuracy']+metrics['f1'])/2 if t=='mrpc' else metrics['accuracy']
                    assert abs(value['tasks'][t]['primary_score']-primary)<1e-12
                    row={'seed':seed,'category':category,'method':method,'task':t,'primary':primary,**metrics}
                    rows.append(row);per.append(row)
                    if category=='static':fitrows.append({'seed':seed,'method':method,'task':t,'aligned_fit_hidden_mse':value['aligned_fit_hidden_mse'][t]})
                    else:
                        own=pred['expert_logits__'+t][tasks.index(t)].argmax(-1)==labels
                        correct=guessed==labels
                        help_=float((correct & ~own).mean());harm=float((~correct & own).mean())
                        assert abs(help_-value['impact_vs_own'][t]['helps_fraction'])<1e-12
                        assert abs(harm-value['impact_vs_own'][t]['harms_fraction'])<1e-12
                        if method=='oracle':assert np.all(correct | ~(pred['expert_logits__'+t].argmax(-1)==labels).any(0))
                        routes.append({'seed':seed,'method':method,'task':t,'helps_fraction':help_,'harms_fraction':harm})
                audit.append({'seed':seed,'category':category,'method':method,
                              'macro_primary':statistics.mean(r['primary'] for r in per),
                              'mean_accuracy':statistics.mean(r['accuracy'] for r in per),
                              'worst_accuracy':min(r['accuracy'] for r in per)})
    aggregate={}
    for method in sorted({r['method'] for r in audit}):
        values=[r for r in audit if r['method']==method]
        aggregate[method]={k:{'mean':statistics.mean(r[k] for r in values),'sample_sd':statistics.stdev(r[k] for r in values)}
                           for k in ['macro_primary','mean_accuracy','worst_accuracy']}
    paired=[]
    for seed in seeds:
        by={r['method']:r for r in audit if r['seed']==seed}
        paired.append({'seed':seed,
            'ce_hidden_minus_ce':by['layer_ce_hidden']['macro_primary']-by['layer_ce_fp32']['macro_primary'],
            'ce_hidden_minus_shuffled':by['layer_ce_hidden']['macro_primary']-by['layer_ce_shuffled']['macro_primary'],
            'layer_hidden_minus_global':by['layer_hidden']['macro_primary']-by['global_hidden']['macro_primary'],
            'expert_router_minus_task_id_accuracy':by['expert_router']['mean_accuracy']-by['task_id']['mean_accuracy'],
            'oracle_minus_task_id_accuracy':by['oracle']['mean_accuracy']-by['task_id']['mean_accuracy']})
    summary={'state':'passed','aggregate':aggregate,'per_seed':audit,'paired':paired,
             'partition_sha256':cfg['partition_manifest_sha256'],
             'audit':'Saved logits, labels, routes, helps/harms and oracle accuracy ceiling independently checked.',
             'scope':'Exploratory extension on the previously inspected v3 final partition; one model, four tasks, three training seeds.'}
    prior=Path(__file__).resolve().parent.parent/'artifacts/recovered-partial-2026-11-09/evidence/per-seed.csv'
    if prior.exists():
        with prior.open() as f: previous=list(csv.DictReader(f))
        indexed={(int(r['seed']),r['method']):r for r in previous}
        differences=[]
        for row in audit:
            old=indexed[(row['seed'],row['method'])]
            differences.append({'seed':row['seed'],'method':row['method'],**{
                k:row[k]-float(old[k]) for k in ['macro_primary','mean_accuracy','worst_accuracy']}})
        summary['repeat_check']={'execution_date':'2026-11-11','previous_csv_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),
            'max_absolute_metric_difference':max(abs(r[k]) for r in differences for k in ['macro_primary','mean_accuracy','worst_accuracy']),
            'differences':differences,'scope':'Same teachers and training seeds; recovery rerun does not increase n beyond three.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name,values in [('per-task.csv',rows),('per-seed.csv',audit),('hidden-fit.csv',fitrows),('routing-impact.csv',routes),('paired.csv',paired)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def chart(methods,labels,name,title,metrics=('macro_primary','worst_accuracy'),subtitles=('Mean native task score','Worst-task accuracy')):
        fig,axes=plt.subplots(1,2,figsize=(11.8,4.7))
        colors=['#526d82','#248a74','#d98a34','#aa4f62','#8a8a8a']
        for ax,key,subtitle in zip(axes,metrics,subtitles):
            means=[aggregate[m][key]['mean'] for m in methods];sd=[aggregate[m][key]['sample_sd'] for m in methods]
            ax.bar(range(len(methods)),means,yerr=sd,capsize=3,color=colors[:len(methods)],alpha=.9)
            for i,m in enumerate(methods):ax.scatter([i-.10,i,i+.10],[r[key] for r in audit if r['method']==m],s=16,c='black',zorder=4)
            ax.set_xticks(range(len(methods)),labels,rotation=22,ha='right');ax.set_ylim(0,1);ax.set_title(subtitle);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle(title,fontsize=13)
        fig.text(.5,.015,'Mean +/- sample SD; three training seeds on one fixed, previously explored partition. Dots: individual seeds.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(out/name,dpi=180);plt.close(fig)
    chart(['layer_ce_fp32','layer_hidden','layer_ce_hidden','layer_ce_shuffled','global_hidden'],
          ['CE FP32','Hidden\n(layer)','CE + hidden','CE + shuffled','Hidden\n(global)'],'activation-controls.png',
          'Direct hidden-response controls: five or 97 trainable scalars, all FP32')
    chart(['task_id','source_router','expert_router','random','oracle'],
          ['Task ID','Source router','Expert router','Random','Oracle\n(final GT)'],'routing-controls.png',
          'Task-aware expert selection: oracle is an analysis-only accuracy ceiling',
          ('mean_accuracy','worst_accuracy'),('Mean task accuracy','Worst-task accuracy'))
    fig,ax=plt.subplots(figsize=(7,4.4))
    for method,label,color in [('source_router','Source router','#526d82'),('expert_router','Expert router','#248a74'),('oracle','Oracle (GT)','#8a8a8a')]:
        values=[r for r in routes if r['method']==method]
        i=['source_router','expert_router','oracle'].index(method)
        help_=statistics.mean(r['helps_fraction'] for r in values)*100;harm=statistics.mean(r['harms_fraction'] for r in values)*100
        ax.bar(i,help_,color=color);ax.bar(i,-harm,color=color,alpha=.45)
        ax.text(i,help_+.4,f'+{help_:.2f}',ha='center');ax.text(i,-harm-.7,f'-{harm:.2f}',ha='center',va='top')
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(range(3),['Source router','Expert router','Oracle (GT)'])
    ax.set_ylabel('Fraction helped / harmed vs own expert (%)');ax.set_title('Routing changes: average over tasks and training seeds')
    ax.margins(y=.25);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(out/'routing-impact.png',dpi=180);plt.close(fig)
    # Portable vector diagram for the PPT author; no model or dataset values invented.
    fig,ax=plt.subplots(figsize=(11,3.5));ax.set_xlim(0,11);ax.set_ylim(0,3.5);ax.axis('off')
    boxes=[(.2,1.15,2.,1.1,'Frozen shared\nbackbone'),(3.,.2,2.,2.9,'Four frozen task\nupdates and heads\n\nDelta W = (alpha/r) BA'),
           (6.,1.15,2.,1.1,'Train only\nmixture + gain'),(8.8,1.15,2.,1.1,'Materialized merge\n+ known task head')]
    for i,(x,y,w,h,label) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor=['#e4edf3','#e4edf3','#d9eee7','#e4edf3'][i],edgecolor='#526d82'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)
    for a,b in [(2.25,2.9),(5.05,5.9),(8.05,8.7)]:ax.annotate('',xy=(b,1.7),xytext=(a,1.7),arrowprops={'arrowstyle':'->','color':'#526d82','lw':2})
    ax.set_title('Static merging pipeline: fitting capacity is separate from deployment storage',fontsize=13)
    fig.tight_layout();fig.savefig(out/'pipeline.png',dpi=180);fig.savefig(out/'pipeline.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11.5,2.5));ax.set_xlim(0,11.5);ax.set_ylim(0,2.5);ax.axis('off')
    for i,(title,count) in enumerate([('Teacher train','up to 4096/task'),('Teacher select','128/task'),('Merge fit','128/task'),('Merge select','128/task'),('Internal final','256/task')]):
        x=.12+i*2.3
        ax.add_patch(FancyBboxPatch((x,.65),1.95,1.0,boxstyle='round,pad=.03',facecolor='#e4edf3' if i!=4 else '#d9eee7',edgecolor='#526d82'))
        ax.text(x+.975,1.15,title+'\n'+count,ha='center',va='center',fontsize=10)
        if i<4:ax.annotate('',xy=(x+2.25,1.15),xytext=(x+2.0,1.15),arrowprops={'arrowstyle':'->','color':'#526d82'})
    ax.set_title('Five content-grouped roles from the original GLUE train split',fontsize=13)
    ax.text(5.75,.15,'Boxes are disjoint roles. Arrows show process order, not reuse of examples. Three training seeds share one partition.',ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(out/'protocol.png',dpi=180);fig.savefig(out/'protocol.svg');plt.close(fig)
    v3=read(study/'evidence/summary.json') if (study/'evidence/summary.json').exists() else read(study/'export/summary.json')
    report=Path(__file__).resolve().parent
    def fmt(v):return f"{v['mean']:.4f} $\\pm$ {v['sample_sd']:.4f}"
    def table(filename,methods,names,source,caption):
        lines=['\\begin{table}[H]\\centering\\small','\\caption{'+caption+'}',
               '\\begin{tabular}{lrrr}\\toprule','Method & Native score & Mean accuracy & Worst accuracy\\\\\\midrule']
        for m,label in zip(methods,names):lines.append(label+' & '+' & '.join(fmt(source[m][k]) for k in ['macro_primary','mean_accuracy','worst_accuracy'])+'\\\\')
        lines+=['\\bottomrule\\end{tabular}\\end{table}']
        (report/filename).write_text('\n'.join(lines)+'\n')
    table('table-v3.tex',['individual','average','task_arithmetic','ties','dare','uniform','layer_ce','layer_kl'],
          ['Individual','Average','Task Arithmetic','TIES (our variant)','DARE+TA','Uniform, selected gain','Layer CE (v3 BF16 fit)','Layer KL (v3 BF16 fit)'],v3['methods'],
          'Frozen protocol-v3 results. Three training seeds; mean and sample SD. Native score is the macro task-specific metric, not official GLUE.')
    table('table-activation.tex',['layer_ce_fp32','layer_hidden','layer_ce_hidden','layer_ce_shuffled','global_hidden'],
          ['Layer CE (FP32)','Layer hidden','Layer CE + hidden','Layer CE + shuffled','Global hidden'],aggregate,
          'Exploratory direct-activation controls. All FP32. Layer methods have 97 scalars; global has five.')
    table('table-routing.tex',['task_id','source_router','expert_router','random','oracle'],
          ['Task ID / own expert','Source router','Expert router','Random','Oracle (final GT)'],aggregate,
          'Exploratory task-aware routing. Oracle is a GT-assisted accuracy ceiling; native scores are not claimed to be upper bounds.')
    ood=read(report.parent/'artifacts/transfer-check-aggregate-2026-10-25.json')
    lines=['\\begin{table}[H]\\centering\\small',
           '\\caption{Historical protocol-v2 fixed transfer check: accuracy mean and sample SD across three training seeds; 512 examples per target.}',
           '\\begin{tabular}{lrr}\\toprule','Method & IMDb & PAWS\\\\\\midrule']
    for key,label in [('individual','Individual'),('average','Average'),('uniform_gain2','Uniform gain 2'),('layer_ce_gt','Layer CE'),('layer_kl_teacher_selected','Layer KL (teacher-selected)')]:
        lines.append(label+' & '+fmt(ood[key]['imdb']['accuracy'])+' & '+fmt(ood[key]['paws']['accuracy'])+'\\\\')
    lines+=['\\bottomrule\\end{tabular}\\end{table}'];(report/'table-ood.tex').write_text('\n'.join(lines)+'\n')
    def paired_text(key):
        values=[r[key] for r in paired];return statistics.mean(values),sum(v>0 for v in values)
    real,wins=paired_text('ce_hidden_minus_ce');shuffle,swins=paired_text('ce_hidden_minus_shuffled')
    routing,rwins=paired_text('expert_router_minus_task_id_accuracy');ceiling,_=paired_text('oracle_minus_task_id_accuracy')
    activation=(f'The paired native-score difference of CE+hidden minus same-precision CE is {real:+.6f}, positive in {wins}/3 seeds. '
                f'Against the shuffled-response control the mean difference is {shuffle:+.6f}, positive in {swins}/3 seeds. '
                + ('The aligned response penalty does not add task-quality benefit over CE in this setting. ' if real<=0 else 'The positive mean effect remains exploratory. ')
                +
                'These descriptive comparisons isolate the selected response-pairing and capacity controls; the sample is too small and the evaluation history too exploratory for a confirmatory claim.')
    routing_text=(f'The expert router changes mean accuracy relative to task-ID selection by {routing*100:+.3f} percentage points, improving in {rwins}/3 seeds. '
                  f'The GT oracle increases mean accuracy by {ceiling*100:.3f} points over task ID. '
                  'This gap measures bank-level headroom under impossible label access, not a realizable routing gain. '
                  + ('Our performance-oriented router fails to realize that headroom and is worse than direct task-ID selection. ' if routing<0 else 'Any observed router benefit remains specific to this exploratory protocol. '))
    macros={'AbstractExtension':f'The exploratory hidden-response and routing controls give paired effects of {real:+.4f} native score versus FP32 CE and {routing*100:+.2f} accuracy points versus task-ID routing, respectively.',
            'ActivationConclusion':activation,'RouterConclusion':routing_text,
            'DiscussionExtension':'The extension results further separate input-response matching from the distribution-preserving shuffle control, and expert identity prediction from actual help or harm relative to task-ID selection.',
            'ConclusionExtension':f'The direct hidden and task-aware routing extensions report paired changes of {real:+.4f} native score and {routing*100:+.2f} accuracy points against their respective controls, with the stated exploratory scope.'}
    (report/'generated-results.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    print(json.dumps({'state':'passed','summary':aggregate,'paired':paired,'activation_text':activation,'routing_text':routing_text}))


if __name__=='__main__':main()
