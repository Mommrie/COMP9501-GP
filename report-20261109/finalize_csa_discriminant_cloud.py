"""Audit discriminative CSA controls, compile the integrated report and export."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time,traceback,zipfile


def main():
    root=Path('/content/COMP9501-GP');report=root/'report-20261109';source=root/'outputs/csa-discriminant-20261111'
    artifact=root/'artifacts/csa-discriminant-2026-11-11';status=root/'outputs/csa-discriminant-materials-status.json';state={'state':'running'}
    env={**os.environ,'USE_TF':'0','USE_FLAX':'0','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    def save(stage):state['current']=stage;status.write_text(json.dumps(state,indent=2))
    def run(stage,args,cwd=root):
        save(stage);log=root/'outputs'/('csa-discriminant-'+stage+'.log')
        with log.open('w') as stream:subprocess.run([str(v) for v in args],cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    try:
        assert json.loads((source/'status.json').read_text())['state']=='passed'
        for name,sha in json.loads((source/'source-sha256.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
        run('audit',[sys.executable,'code/summarize_csa_discriminant.py','--input',source,'--bridge',root/'outputs/csa-bridge-20261111','--previous',root/'outputs/activation-sensitivity-20261111','--output',artifact])
        s=json.loads((artifact/'summary.json').read_text());methods=s['methods'];own=methods['task_id']['reference'];c=methods['csa2x8']
        lines=[f"At the fixed CSA $2\\times8$ setting, direct LDA has native score {c['direct_lda']['native']['mean']:.4f}, versus {c['previous_logistic']['native']['mean']:.4f} for the earlier logistic router and {own['native']['mean']:.4f} for task-ID selection."]
        for family,label in [('task_prior','task-prior gate'),('benefit_lda','benefit-LDA gate')]:
            v=c[family];lines.append(f"The {label} has native score {v['native']['mean']:.4f}; it chooses always-own in {v['always_own_seeds']}/3 seeds and makes nonzero final switches in {v['nonzero_switch_seeds']}/3. Mean switched, helped and harmed fractions are {v['switch_fraction']['mean']*100:.3f}\\%, {v['helps']['mean']*100:.3f}\\%, and {v['harms']['mean']*100:.3f}\\%, respectively. Its macro-accuracy change versus task ID is {v['accuracy_delta']['mean']*100:+.3f} percentage points.")
        lines.append('These outcomes distinguish discriminative fitting, conservative fallback and actual useful switching. Matching the reference with zero switches is baseline preservation rather than evidence of learned routing benefit. Selection constraints alone do not establish final safety or a statistically significant improvement.')
        (report/'csa-discriminant-findings.tex').write_text('\n\n'.join(lines)+'\n');(report/'table-csa-discriminant.tex').write_bytes((artifact/'table-csa-discriminant.tex').read_bytes())
        checks=[];count=0
        for folder,pattern in [('code','test_math.py'),('code','test_protocol.py'),('code','test_independent_protocol.py'),('code','test_activation_extension.py'),('code/coefficient_merge','test_core.py'),('code/merge_study','test_engine.py'),('code','test_robustness_controls.py'),('code','test_csa_bridge.py'),('code','test_csa_discriminant.py'),('scripts','test_colab_bundle.py'),('scripts','test_team_notebook.py'),('code','test_integration.py')]:
            save('test:'+pattern);proc=subprocess.run([sys.executable,'-m','unittest','discover','-s',folder,'-p',pattern],cwd=root,env=env,capture_output=True,text=True)
            output=proc.stdout+proc.stderr;assert proc.returncode==0,output;count+=sum(map(int,re.findall(r'Ran (\d+) tests?',output)));checks.append({'folder':folder,'pattern':pattern,'returncode':proc.returncode,'output':output})
        assert count==37,count;run('notebook-check',[sys.executable,'scripts/build_team_notebook.py','--check'])
        validation={'state':'passed','test_count':count,'checks':checks,'tested_source_sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in list((root/'code').rglob('*.py'))+[root/'scripts/build_team_notebook.py',root/'part1-reproduction-20261025/code/csa_tensor.py']}}
        (report/'code-validation.json').write_text(json.dumps(validation,indent=2));(artifact/'code-validation.json').write_text(json.dumps(validation,indent=2))
        run('latex',[root/'outputs/report-tools/tectonic','--keep-logs','report.tex'],cwd=report)
        assert 'Overfull' not in (report/'report.log').read_text(errors='replace'),'Overfull report layout'
        render=root/'outputs/csa-discriminant-render';render.mkdir(exist_ok=False)
        run('pdftoppm',['pdftoppm','-r','90','-png',report/'report.pdf',render/'page']);run('pdftotext',['pdftotext','-layout',report/'report.pdf',render/'report-text.txt'])
        pages=sorted(render.glob('page-*.png'));assert len(pages)>=18
        from PIL import Image,ImageDraw
        for start in range(0,len(pages),6):
            sheet=Image.new('RGB',(960,720),'#dddddd');draw=ImageDraw.Draw(sheet)
            for i,p in enumerate(pages[start:start+6]):
                im=Image.open(p).convert('RGB');im.thumbnail((310,330));x=(i%3)*320+(320-im.width)//2;y=(i//3)*360+18;sheet.paste(im,(x,y));draw.text((x,y-15),str(start+i+1),fill='black')
            sheet.save(render/f'contact-{start//6+1}.png')
        sha=hashlib.sha256((report/'report.pdf').read_bytes()).hexdigest()
        build={'state':'compiled-and-rendered','report_version':'csa-discriminant','pages':len(pages),'pdf_sha256':sha,'previous_published_pdf_sha256':'1b62e81e49e530fa626a53d1beadf6976dce10e9a9b8662b8a9c658f14707ea6','layout_overfull_warnings':0,'compiler':json.loads((root/'outputs/report-tools/ready.json').read_text())}
        (report/'build-verification.json').write_text(json.dumps(build,indent=2));(report/'visual-review.json').write_text(json.dumps({'state':'review-required','pdf_sha256':sha},indent=2))
        backup=Path('/content/COMP9501-Part2-CSA-discriminant-state-20261111.zip')
        with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(source.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(root).as_posix())
        (artifact/'state-backup.json').write_text(json.dumps({'name':backup.name,'bytes':backup.stat().st_size,'sha256':hashlib.sha256(backup.read_bytes()).hexdigest(),'scope':'All candidate discriminants/scalers and selection/final evidence; restore prior CSA response/projection backup separately.'},indent=2))
        (artifact/'README.md').write_text('# Discriminative CSA and task-ID residual controls\n\nFrozen regularized LDA, prior margins and benefit gates on all prior response representations. See the [protocol](../../report-20261109/CSA-DISCRIMINANT-PROTOCOL.zh-CN.md). Every selection candidate and selected final route is reloaded and audited. Summary/CSV include actual switch/help/harm and always-own counts. Full learned discriminants are backed up separately. All-expert offline inputs, no serving-speed claim; previously explored internal final.\n')
        (artifact/'manifest.json').write_text(json.dumps({'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(artifact.iterdir()) if p.is_file() and p.name!='manifest.json'}},indent=2))
        package=Path('/content/COMP9501-Part2-CSA-discriminant-materials-20261111.zip');review=Path('/content/COMP9501-Part2-CSA-discriminant-review-20261111.zip')
        with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(list(artifact.iterdir())+list(report.iterdir())):
                if p.is_file() and p.suffix not in {'.aux','.log','.out','.toc'}:z.write(p,p.relative_to(root).as_posix())
        with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(render.iterdir()):z.write(p,p.name)
            z.write(report/'report.pdf','report.pdf')
        metadata={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [package,review,backup]}
        Path('/content/COMP9501-csa-discriminant-ready.json').write_text(json.dumps(metadata,indent=2));state.update(state='exported',pages=len(pages),archives=metadata);save(None)
    except Exception as e:
        state.update(state='error',error=str(e),traceback=traceback.format_exc());save('error');raise


if __name__=='__main__':main()
