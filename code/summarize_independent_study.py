"""Audit and export the finished cloud study; never choose new hyperparameters."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run this export in the Colab study environment')
    from independent_protocol import validate_manifest
    p = argparse.ArgumentParser()
    p.add_argument('--study', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    study, out = Path(args.study), Path(args.output)
    out.mkdir(exist_ok=False)
    read = lambda p: json.loads(p.read_text())
    state = read(study/'status.json')
    baselines = read(study/'weight-baselines/summary.json')
    assert state['state']==baselines['state']=='passed'
    cfg = read(study/'locked-config.json')
    manifest = validate_manifest(read(study/'partitions.json'))
    assert hashlib.sha256((study/'partitions.json').read_bytes()).hexdigest()==cfg['partition_manifest_sha256']
    all_rows, seed_rows, results, audit = [], [], {}, []
    for seed in cfg['seeds']:
        coeff = read(study/f'coefficients-s{seed}/summary.json')
        assert coeff['state']=='passed'
        selected = read(study/f'coefficients-s{seed}/selection.json')
        assert selected['locked_before_validation']
        assert coeff['fits']['layer_ce']['learnable_parameters']==coeff['fits']['layer_kl']['learnable_parameters']==97
        for task in cfg['tasks']:
            old = coeff['uniform_selection']['1.0']['tasks'][task]['metrics']
            direct = baselines['seeds'][str(seed)]['selection']['average']['tasks'][task]['metrics']
            assert old == direct, 'Uniform gain=1 and Average selection evaluations differ'
        methods = {'individual': coeff['individual'],
                   **{m:v['tasks'] for m,v in coeff['validation'].items()},
                   **{m:v['tasks'] for m,v in baselines['seeds'][str(seed)]['validation'].items()}}
        for task in cfg['tasks']:
            actual = read(study/f'bank-s{seed}/{task}/data-manifest.json')
            parts = manifest['tasks'][task]['partitions']
            assert actual['train']['sample_ids']==parts['train']['sample_ids']
            assert actual['selection']['sample_ids']==parts['calibration']['sample_ids']
            trained = read(study/f'bank-s{seed}/{task}/run.json')
            assert trained['config']['partition_manifest_sha256']==cfg['partition_manifest_sha256']
            assert trained['resolved_model_revision']==cfg['model_revision']
            split = read(study/f'coefficients-s{seed}/split-ids.json')[task]
            assert split['fit']==parts['coefficient_fit']['sample_ids']
            assert split['selection']==parts['coefficient_selection']['sample_ids']
        for method, values in methods.items():
            primary = [values[t]['primary_score'] for t in cfg['tasks']]
            accuracy = [values[t]['metrics']['accuracy'] for t in cfg['tasks']]
            drop = [methods['individual'][t]['metrics']['accuracy']-values[t]['metrics']['accuracy'] for t in cfg['tasks']]
            row = {'seed': seed, 'method': method, 'macro_primary': statistics.mean(primary),
                   'mean_accuracy': statistics.mean(accuracy), 'worst_accuracy': min(accuracy),
                   'mean_accuracy_drop': statistics.mean(drop), 'maximum_accuracy_drop': max(drop)}
            seed_rows.append(row)
            results.setdefault(method, []).append(row)
            for task in cfg['tasks']:
                assert values[task]['samples']==len(manifest['tasks'][task]['partitions']['validation']['sample_ids'])
                all_rows.append({'seed':seed,'method':method,'task':task,'primary':values[task]['primary_score'],
                                 **values[task]['metrics'],'accuracy_drop':methods['individual'][task]['metrics']['accuracy']-values[task]['metrics']['accuracy']})
        audit.append({'seed':seed,'teacher_data_and_model_revision_match':True,'coefficient_split_match':True,
                      'equal_parameter_count':97,'ce_step':coeff['fits']['layer_ce']['selected']['step'],
                      'kl_step':coeff['fits']['layer_kl']['selected']['step']})
    summary = {method:{key:{'mean':statistics.mean(r[key] for r in rows),
                             'sample_sd':statistics.stdev(r[key] for r in rows)}
                       for key in ['macro_primary','mean_accuracy','worst_accuracy','mean_accuracy_drop','maximum_accuracy_drop']}
               for method,rows in results.items()}
    paired = [{key:results['layer_kl'][i][key]-results['layer_ce'][i][key]
               for key in ['macro_primary','mean_accuracy','worst_accuracy']} | {'seed':seed}
              for i,seed in enumerate(cfg['seeds'])]
    write = lambda name,value:(out/name).write_text(json.dumps(value,indent=2)+'\n')
    write('summary.json',{'methods':summary,'per_seed':seed_rows,'paired_kl_minus_ce':paired,
                          'audit':audit,'config':cfg,'partition_sha256':cfg['partition_manifest_sha256'],
                          'scope':'Three training seeds on one fixed internally isolated partition of previously explored GLUE tasks; not pristine external evaluation.'})
    for name,rows in [('per-task.csv',all_rows),('per-seed.csv',seed_rows)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    methods=['individual','average','task_arithmetic','ties','dare','uniform','layer_ce','layer_kl']
    labels=['Individual','Average','TA','TIES','DARE+TA','Uniform\nselected','Layer CE','Layer KL']
    colors=['#526d82']*6+['#d98a34','#248a74']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,key,title in zip(axes,['macro_primary','worst_accuracy'],['Mean native task score','Worst-task accuracy']):
        means=[summary[m][key]['mean'] for m in methods]
        sd=[summary[m][key]['sample_sd'] for m in methods]
        ax.bar(range(len(methods)),means,yerr=sd,color=colors,capsize=3,alpha=.85)
        for i,m in enumerate(methods):
            ax.scatter([i-.12,i,i+.12],[r[key] for r in results[m]],s=15,color='black',zorder=4)
        ax.set_xticks(range(len(methods)),labels,rotation=35,ha='right');ax.set_ylim(0,1)
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Fresh-teacher protocol: Qwen2.5-0.5B, four GLUE tasks')
    fig.text(.5,.015,'Bars: mean +/- sample SD over training seeds 42/43/44. Dots: individual seeds. One fixed partition; known task heads.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(out/'main-comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for item in paired:
        ax.plot([0,1],[results['layer_ce'][cfg['seeds'].index(item['seed'])]['macro_primary'],results['layer_kl'][cfg['seeds'].index(item['seed'])]['macro_primary']],marker='o',label=f"Seed {item['seed']}: delta={item['macro_primary']:+.4f}")
    ax.set_xticks([0,1],['Layer CE (97 parameters)','Layer KL (97 parameters)']);ax.set_ylabel('Mean native task score')
    ax.set_title('Matched-capacity paired comparison');ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.text(.5,.015,'Same inputs, updates and GT checkpoint-selection access; no significance claim from n=3.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(out/'paired-ce-kl.png',dpi=180);plt.close(fig)
    matrix=np.array([[statistics.mean(r['accuracy'] for r in all_rows if r['method']==m and r['task']==t) for t in cfg['tasks']] for m in methods])
    fig,ax=plt.subplots(figsize=(7,5));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu')
    for i in range(len(methods)):
        for j in range(len(cfg['tasks'])):ax.text(j,i,f'{matrix[i,j]*100:.1f}',ha='center',va='center',color='white' if matrix[i,j]>.75 else 'black')
    ax.set_xticks(range(4),[t.upper() for t in cfg['tasks']]);ax.set_yticks(range(len(methods)),[s.replace('\n',' ') for s in labels])
    ax.set_title('Per-task accuracy (%), mean of three training seeds');fig.colorbar(im,ax=ax,label='Accuracy',shrink=.8)
    fig.tight_layout();fig.savefig(out/'task-accuracy.png',dpi=180);plt.close(fig)
    write('export-ready.json',{'state':'passed','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(out.iterdir()) if p.is_file()}})
    print(json.dumps({'summary':summary,'paired':paired,'audit':audit}))


if __name__=='__main__':
    main()
