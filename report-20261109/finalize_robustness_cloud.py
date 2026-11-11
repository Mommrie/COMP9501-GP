"""Finish the second follow-up: full tests, prediction audit, LaTeX and exports."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


def main():
    root=Path('/content/COMP9501-GP');report=root/'report-20261109';status=root/'outputs/robustness-materials-status.json'
    state={'state':'waiting-results'}
    def save(current):state['current']=current;status.write_text(json.dumps(state,indent=2))
    env=dict(os.environ,USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    def run(name,args,cwd=root):
        save(name);log=root/'outputs'/('robustness-'+name+'.log')
        with log.open('w') as stream:subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        return log
    try:
        save('waiting');deadline=time.monotonic()+4*3600
        while True:
            p=root/'outputs/robustness-pipeline-status.json';value=json.loads(p.read_text())
            if value['state']=='error':raise RuntimeError(value)
            if value['state']=='results-passed':break
            if time.monotonic()>deadline:raise TimeoutError('Follow-up results')
            time.sleep(15)
        for name in ['activation-sensitivity-20261111','robustness-controls-20261111']:
            for path,digest in json.loads((root/'outputs'/name/'source-sha256.json').read_text()).items():
                assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
        save('full-code-checks');checks=[];count=0
        for folder,pattern in [('code','test_math.py'),('code','test_protocol.py'),('code','test_independent_protocol.py'),
                               ('code','test_activation_extension.py'),('code/coefficient_merge','test_core.py'),('code/merge_study','test_engine.py'),
                               ('code','test_robustness_controls.py'),('scripts','test_colab_bundle.py'),('scripts','test_team_notebook.py'),('code','test_integration.py')]:
            p=subprocess.run([sys.executable,'-m','unittest','discover','-s',folder,'-p',pattern],cwd=root,env=env,capture_output=True,text=True)
            output=p.stdout+p.stderr;checks.append({'folder':folder,'pattern':pattern,'returncode':p.returncode,'output':output})
            assert p.returncode==0,output
            count+=sum(map(int,re.findall(r'Ran (\d+) tests?',output)))
        p=subprocess.run([sys.executable,'scripts/build_team_notebook.py','--check'],cwd=root,env=env,capture_output=True,text=True)
        assert p.returncode==0,p.stdout+p.stderr
        checks.append({'command':'scripts/build_team_notebook.py --check','returncode':0,'output':p.stdout+p.stderr})
        assert count==28,count
        validation={'state':'passed','test_count':count,'checks':checks,
                    'tested_source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((root/'code').rglob('*.py'))+[root/'scripts/build_team_notebook.py'] if '__pycache__' not in p.parts}}
        (root/'outputs/robustness-final-validation.json').write_text(json.dumps(validation,indent=2))
        artifact=root/'artifacts/robustness-v2-2026-11-11'
        run('audit-plots',[sys.executable,'code/summarize_robustness.py','--previous','outputs/activation-strengthening-20261109',
            '--sensitivity','outputs/activation-sensitivity-20261111','--controls','outputs/robustness-controls-20261111','--output',artifact])
        texlog=run('latex',[root/'outputs/report-tools/tectonic','--keep-logs','report.tex'],cwd=report)
        assert 'Overfull' not in texlog.read_text(),texlog.read_text()
        pdf=report/'report.pdf';text=root/'outputs/robustness-report-text.txt'
        run('pdftotext',['pdftotext','-layout',pdf,text]);assert '??' not in text.read_text()
        render=root/'outputs/robustness-report-render';render.mkdir(exist_ok=False)
        run('render',['pdftoppm','-scale-to','1250','-png',pdf,render/'page'])
        from PIL import Image,ImageDraw
        pages=sorted(render.glob('page-*.png'),key=lambda p:int(p.stem.split('-')[-1]))
        for start in range(0,len(pages),6):
            sheet=Image.new('RGB',(900,1980),'#e7e9ed');draw=ImageDraw.Draw(sheet)
            for i,path in enumerate(pages[start:start+6]):
                im=Image.open(path).convert('RGB');im.thumbnail((420,610));x=(i%2)*450+15;y=(i//2)*660+30
                sheet.paste(im,(x,y));draw.text((x,y-20),'Page '+str(start+i+1),fill='black')
            sheet.save(render/f'contact-{start//6+1}.png')
        digest=hashlib.sha256(pdf.read_bytes()).hexdigest()
        build={'state':'compiled-and-rendered','report_version':'robustness-v2','pages':len(pages),'pdf_sha256':digest,
               'previous_published_pdf_sha256':'63e4ffed2e97006e78da282aeb8810c3c0718decb27d4d684ffbdf75de915e33',
               'layout_overfull_warnings':0,'compiler':json.loads((root/'outputs/report-tools/ready.json').read_text()),
               'log_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'outputs').glob('robustness-*.log')},
               'note':'All final pages still require visual inspection before publication.'}
        (report/'build-verification.json').write_text(json.dumps(build,indent=2))
        (report/'visual-review.json').write_text(json.dumps({'state':'review-required','pdf_sha256':digest},indent=2))
        (report/'code-validation.json').write_text(json.dumps(validation,indent=2))
        main=Path('/content/COMP9501-Part2-robustness-20261111.zip');review=Path('/content/COMP9501-Part2-robustness-review-20261111.zip')
        with zipfile.ZipFile(main,'w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(list(report.iterdir())+list(artifact.iterdir())):
                if path.is_file() and path.suffix not in {'.aux','.out','.log','.toc','.synctex'}:z.write(path,path.relative_to(root).as_posix())
        with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(render.iterdir()):z.write(path,path.name)
            z.write(pdf,'report.pdf');z.write(text,'report-text.txt')
        metadata={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [main,review]}
        Path('/content/COMP9501-robustness-ready.json').write_text(json.dumps(metadata,indent=2))
        state.update(state='exported',pages=len(pages),archives=metadata);save(None)
    except Exception as error:
        state.update(state='error',error=str(error),traceback=traceback.format_exc());save('error');raise


if __name__=='__main__':main()
