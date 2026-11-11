"""Audit direct CSA transfer, compile the integrated report and export backups."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time,traceback,zipfile


def main():
    root=Path('/content/COMP9501-GP');report=root/'report-20261109';source=root/'outputs/csa-bridge-20261111'
    artifact=root/'artifacts/csa-bridge-2026-11-11';status=root/'outputs/csa-bridge-materials-status.json';state={'state':'waiting'}
    env={**os.environ,'USE_TF':'0','USE_FLAX':'0','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    def save(stage):state['current']=stage;status.write_text(json.dumps(state,indent=2))
    def run(stage,args,cwd=root):
        save(stage);log=root/'outputs'/('csa-bridge-'+stage+'.log')
        with log.open('w') as stream:subprocess.run([str(v) for v in args],cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        return log
    try:
        deadline=time.monotonic()+3600
        while True:
            current=json.loads((source/'status.json').read_text())
            if current['state']=='passed':break
            if current['state']=='error':raise RuntimeError(current)
            if time.monotonic()>deadline:raise TimeoutError('CSA computation')
            time.sleep(10)
        for name,sha in json.loads((source/'source-sha256.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
        run('audit',[sys.executable,'code/summarize_csa_bridge.py','--input',source,'--previous',root/'outputs/activation-sensitivity-20261111','--output',artifact])
        s=json.loads((artifact/'summary.json').read_text());m=s['representations'];routes=s['routing'];c=m['csa2x8'];p=m['pca16']
        d=(c['storage']['total']-c['storage']['mean'])//(512+4*896);b=m['pca'+str(d)]
        findings=(f"For the illustrative $(r_E,r_H)=(2,8)$ setting, CSA stores {c['storage']['total']*8/1024:.1f} KiB, versus {p['storage']['total']*8/1024:.1f} KiB for equal-latent PCA, a {p['storage']['total']/c['storage']['total']:.2f}-fold reduction in this representation scope. The largest budget-feasible PCA rank is {d}. Internal-final relative squared errors are {c['final_error_mean']:.4f} (CSA), {p['final_error_mean']:.4f} (equal-latent PCA), and {b['final_error_mean']:.4f} (budget PCA). Their native routing scores are {routes['csa2x8']['mean']:.4f}, {routes['pca16']['mean']:.4f}, and {routes['pca'+str(d)]['mean']:.4f}, respectively; raw-delta routing is {routes['raw_delta']['mean']:.4f} and task-ID is {routes['task_id']['mean']:.4f}. These comparisons separate representation compactness, reconstruction fidelity and task utility. The full predeclared grid is available in the evidence CSV; this illustration is not a final-selected best configuration.\n")
        (report/'csa-bridge-findings.tex').write_text(findings)
        (report/'table-csa-bridge.tex').write_bytes((artifact/'table-csa-bridge.tex').read_bytes())
        checks=[];count=0
        for folder,pattern in [('code','test_math.py'),('code','test_protocol.py'),('code','test_independent_protocol.py'),('code','test_activation_extension.py'),('code/coefficient_merge','test_core.py'),('code/merge_study','test_engine.py'),('code','test_robustness_controls.py'),('code','test_csa_bridge.py'),('scripts','test_colab_bundle.py'),('scripts','test_team_notebook.py'),('code','test_integration.py')]:
            save('test:'+pattern);proc=subprocess.run([sys.executable,'-m','unittest','discover','-s',folder,'-p',pattern],cwd=root,env=env,capture_output=True,text=True)
            text=proc.stdout+proc.stderr;assert proc.returncode==0,text;count+=sum(map(int,re.findall(r'Ran (\d+) tests?',text)))
            checks.append({'folder':folder,'pattern':pattern,'returncode':proc.returncode,'output':text})
        assert count==32,count
        run('notebook-check',[sys.executable,'scripts/build_team_notebook.py','--check'])
        validation={'state':'passed','test_count':count,'checks':checks,'tested_source_sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in list((root/'code').rglob('*.py'))+[root/'scripts/build_team_notebook.py',root/'part1-reproduction-20261025/code/csa_tensor.py']}}
        (report/'code-validation.json').write_text(json.dumps(validation,indent=2));(artifact/'code-validation.json').write_text(json.dumps(validation,indent=2))
        run('latex',[root/'outputs/report-tools/tectonic','--keep-logs','report.tex'],cwd=report)
        log=(report/'report.log').read_text(errors='replace');assert 'Overfull' not in log,'Overfull report layout'
        render=root/'outputs/csa-bridge-report-render';render.mkdir(exist_ok=False)
        run('pdftoppm',['pdftoppm','-r','90','-png',report/'report.pdf',render/'page'])
        run('pdftotext',['pdftotext','-layout',report/'report.pdf',render/'report-text.txt'])
        pages=sorted(render.glob('page-*.png'));assert len(pages)>=17,len(pages)
        from PIL import Image,ImageOps,ImageDraw
        for start in range(0,len(pages),6):
            sheet=Image.new('RGB',(960,720),'#dddddd');draw=ImageDraw.Draw(sheet)
            for i,path in enumerate(pages[start:start+6]):
                im=Image.open(path).convert('RGB');im.thumbnail((310,330));x=(i%3)*320+(320-im.width)//2;y=(i//3)*360+18;sheet.paste(im,(x,y));draw.text((x,y-15),str(start+i+1),fill='black')
            sheet.save(render/f'contact-{start//6+1}.png')
        sha=hashlib.sha256((report/'report.pdf').read_bytes()).hexdigest()
        build={'state':'compiled-and-rendered','report_version':'direct-csa-bridge','pages':len(pages),'pdf_sha256':sha,'previous_published_pdf_sha256':'26fae3e18b3f7fc0ecdbbecaceb8cb8ac95cff053f2f5e7ddd2a24c1e3e5c895','layout_overfull_warnings':0,'compiler':json.loads((root/'outputs/report-tools/ready.json').read_text())}
        (report/'build-verification.json').write_text(json.dumps(build,indent=2));(report/'visual-review.json').write_text(json.dumps({'state':'review-required','pdf_sha256':sha},indent=2))
        backup=Path('/content/COMP9501-Part2-CSA-bridge-state-20261111.zip')
        save('state-backup')
        with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(source.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(root).as_posix())
        backup_info={'name':backup.name,'bytes':backup.stat().st_size,'sha256':hashlib.sha256(backup.read_bytes()).hexdigest(),'scope':'Exact response caches and learned encoders/routers; public-data derived features, no model weights or raw text.'}
        (artifact/'state-backup.json').write_text(json.dumps(backup_info,indent=2))
        (artifact/'README.md').write_text('# Direct CSA-to-LoRA response bridge\n\nThe unchanged Part 1 TensorCSA is applied to expert-by-hidden response matrices. See the frozen [protocol](../../report-20261109/CSA-BRIDGE-PROTOCOL.zh-CN.md). Summary/CSV, per-seed predictions, role IDs, selection locks, sources and independent reload audit are included. Mean/basis/code storage and same-latent versus budget PCA are separate comparisons. All-expert features require all expert forwards; this is an offline response-cache diagnostic, not weight compression or deployment acceleration. Complete learned state and response caches are backed up separately; state-backup.json records their checksum.\n')
        (artifact/'manifest.json').write_text(json.dumps({'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(artifact.iterdir()) if p.is_file() and p.name!='manifest.json'}},indent=2))
        package=Path('/content/COMP9501-Part2-CSA-bridge-materials-20261111.zip');review=Path('/content/COMP9501-Part2-CSA-bridge-review-20261111.zip')
        with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(list(artifact.iterdir())+list(report.iterdir())):
                if p.is_file() and p.suffix not in {'.aux','.log','.out','.toc'}:z.write(p,p.relative_to(root).as_posix())
        with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(render.iterdir()):z.write(p,p.name)
            z.write(report/'report.pdf','report.pdf')
        metadata={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [package,review,backup]}
        Path('/content/COMP9501-csa-bridge-ready.json').write_text(json.dumps(metadata,indent=2));state.update(state='exported',pages=len(pages),archives=metadata);save(None)
    except Exception as e:
        state.update(state='error',error=str(e),traceback=traceback.format_exc());save('error');raise


if __name__=='__main__':main()
