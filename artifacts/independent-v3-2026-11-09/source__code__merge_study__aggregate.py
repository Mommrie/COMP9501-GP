"""Aggregate training seeds without treating overlapping task pairs as IID."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean, stdev


def stats(values):
    return {"mean":mean(values),"sd":stdev(values),"values":values}


def main():
    import numpy as np
    from scipy.stats import rankdata
    parser=argparse.ArgumentParser()
    parser.add_argument("--studies",nargs="+",required=True)
    parser.add_argument("--output",required=True,help="JSON output path")
    parser.add_argument("--plot",required=True,help="PNG/PDF output prefix")
    args=parser.parse_args()
    target,plot=Path(args.output),Path(args.plot)
    if any(p.exists() for p in [target,plot.with_suffix('.png'),plot.with_suffix('.pdf')]):
        raise FileExistsError("Output already exists")
    studies=sorted([json.loads(Path(p).read_text(encoding="utf-8")) for p in args.studies],key=lambda s:s['seed'])
    assert [s['seed'] for s in studies]==[42,43,44]
    assert all(s['state']=='passed' and len(s['pairs'])==6 for s in studies)
    assert all(s['config']==studies[0]['config'] for s in studies)
    assert all(s['plan']['candidates']==studies[0]['plan']['candidates'] for s in studies)
    assert all(s['plan']['merge_seed']==42 for s in studies)
    tasks=studies[0]['config']['tasks'];methods=list(studies[0]['validation'])
    result={'seeds':[s['seed'] for s in studies],'training_seed_repeats':3,'unique_task_pairs':6,
        'analysis_environment':{name:importlib.metadata.version(name) for name in ['numpy','scipy','matplotlib']},
        'aggregation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'mask_seed':42,'methods':{},'correlations':{},'pairs':[],
        'source_summaries_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.studies},
        'interpretation':'Mean and sample SD across three training seeds with a fixed dataset split. No p values; pairs overlap and must not be counted as 18 independent observations.',
        'selections':{str(s['seed']):s['selection']['selected'] for s in studies}}
    for method in methods:
        by_task={t:{metric:stats([s['validation'][method]['tasks'][t]['metrics'][metric] for s in studies])
                    for metric in ['accuracy','f1','mcc']} for t in tasks}
        primary=[s['validation'][method]['macro_primary_score'] for s in studies]
        accuracy=[mean(s['validation'][method]['tasks'][t]['metrics']['accuracy'] for t in tasks) for s in studies]
        worst=[min(s['validation'][method]['tasks'][t]['metrics']['accuracy'] for t in tasks) for s in studies]
        result['methods'][method]={'tasks':by_task,'macro_primary_score':stats(primary),
            'mean_accuracy':stats(accuracy),'worst_accuracy':stats(worst),
            'worst_task_by_seed':[min(tasks,key=lambda t:s['validation'][method]['tasks'][t]['metrics']['accuracy']) for s in studies]}
    def correlation(x,y):
        return {'pearson':float(np.corrcoef(x,y)[0,1]),'spearman':float(np.corrcoef(rankdata(x),rankdata(y))[0,1])}
    for signal in ['delta_cosine','raw_cosine','weight_cosine']:
        values=[]
        for s in studies:
            computed=correlation([p[signal] for p in s['pairs']],[p['mean_accuracy_drop'] for p in s['pairs']])
            assert abs(computed['spearman']-s['correlations'][signal]['spearman'])<1e-12
            assert abs(computed['pearson']-s['correlations'][signal]['pearson'])<1e-12
            values.append(computed)
        result['correlations'][signal]={key:stats([v[key] for v in values]) for key in ['pearson','spearman']}
    for index,reference in enumerate(studies[0]['pairs']):
        pair=[s['pairs'][index] for s in studies]
        assert all(p['tasks']==reference['tasks'] for p in pair)
        result['pairs'].append({'tasks':reference['tasks'],
            'same_input_format_family':reference['same_input_format_family'],
            **{key:stats([p[key] for p in pair]) for key in ['delta_cosine','raw_cosine','weight_cosine','mean_accuracy_drop','mean_primary_drop']}})
    result['correlation_of_pair_means']={signal:correlation([p[signal]['mean'] for p in result['pairs']],
        [p['mean_accuracy_drop']['mean'] for p in result['pairs']]) for signal in ['delta_cosine','raw_cosine','weight_cosine']}
    result['per_seed_primary_scores']={str(s['seed']):{m:s['validation'][m]['macro_primary_score'] for m in methods} for s in studies}
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained')
    ax=axes[0]
    for p in result['pairs']:
        color='#c45436' if p['same_input_format_family'] else '#21618c'
        x,y=p['delta_cosine'],p['mean_accuracy_drop']
        ax.errorbar(x['mean'],y['mean']*100,xerr=x['sd'],yerr=y['sd']*100,
                    fmt='s' if p['same_input_format_family'] else 'o',color=color,capsize=3,ms=6)
        ax.annotate('/'.join(t.upper() for t in p['tasks']),(x['mean'],y['mean']*100),
                    xytext=(5,6),textcoords='offset points',fontsize=8)
    ax.axhline(0,color='0.6',lw=.7);ax.grid(alpha=.2);ax.margins(x=.25,y=.3)
    ax.set_xlabel('Delta activation cosine');ax.set_ylabel('Mean pair accuracy loss (percentage points)')
    ax.set_title('Six unique task pairs: mean +/- sample SD\nacross training seeds 42, 43, 44',fontsize=11)
    ax=axes[1];x=np.arange(3);width=.23
    for offset,signal,label,color in [(-1,'delta_cosine','Delta activation','#21618c'),
                                     (0,'raw_cosine','Raw activation','#c45436'),
                                     (1,'weight_cosine','Weight delta','#6f7785')]:
        values=result['correlations'][signal]['spearman']['values']
        ax.bar(x+offset*width,values,width=width,label=label,color=color)
    ax.set_xticks(x);ax.set_xticklabels(['Seed 42','Seed 43','Seed 44']);ax.set_ylim(-1.05,1.05)
    ax.set_ylabel('Spearman rho vs mean pair accuracy loss');ax.axhline(0,color='0.5',lw=.7)
    ax.grid(axis='y',alpha=.2);ax.legend(loc='upper right',fontsize=9)
    ax.set_title('Within-seed correlations (six overlapping pairs)\nNegative rho matches the proposed direction',fontsize=11)
    fig.suptitle('Protocol-v2 replication: three training seeds, fixed data split and DARE mask seed',fontsize=12)
    fig.supxlabel('Error bars are training-seed SD, not confidence intervals. No significance claim.',fontsize=9)
    plot.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(plot.with_suffix('.png'),dpi=180);fig.savefig(plot.with_suffix('.pdf'));plt.close(fig)
    print('THREE_SEED_AGGREGATION_COMPLETE',target)


if __name__=='__main__':main()
