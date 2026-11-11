"""Wait for the fixed study, audit predictions, compile/render report, export evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/content/COMP9501-GP');args=p.parse_args()
    root=Path(args.root);ext=root/'outputs/activation-strengthening-20261109'
    study=root/'outputs/independent-v3-20261109';report=root/'report-20261109'
    result=root/'outputs/course-materials-status.json'
    state={'state':'waiting-for-extension'}
    def publish(current):
        state['current']=current;result.write_text(json.dumps(state,indent=2))
    def run(name,command,cwd=root):
        publish(name)
        with (root/'outputs'/(name+'.log')).open('w') as log:
            subprocess.run(list(map(str,command)),cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        publish('waiting')
        deadline=time.monotonic()+4*3600
        while True:
            path=ext/'status.json'
            if path.exists():
                status=json.loads(path.read_text())
                if status['state']=='passed':break
                if status['state']=='error':raise RuntimeError(status.get('error'))
            if time.monotonic()>deadline:raise TimeoutError('Study did not complete within four hours')
            time.sleep(15)
        artifact=root/'artifacts/activation-strengthening-2026-11-11'
        for name,digest in json.loads((ext/'source-sha256.json').read_text()).items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        run('course-audit-and-plots',[sys.executable,report/'build_materials.py','--study',study,'--extension',ext,'--artifacts',artifact])
        tool=root/'outputs/report-tools/tectonic'
        assert tool.exists(),'Report compiler is not installed'
        run('course-latex',[tool,'--keep-logs','report.tex'],cwd=report)
        pdf=report/'report.pdf';assert pdf.stat().st_size>30000
        run('course-pdfinfo',['pdfinfo',pdf])
        textfile=root/'outputs/course-report-text.txt'
        run('course-pdftotext',['pdftotext','-layout',pdf,textfile])
        text=textfile.read_text()
        assert '??' not in text and 'Conclusion' in text and '0.7132' in text
        render=root/'outputs/course-render';render.mkdir(exist_ok=False)
        run('course-render',['pdftoppm','-scale-to','1250','-png',pdf,render/'page'])
        from PIL import Image,ImageOps,ImageDraw
        pages=sorted(render.glob('page-*.png'),key=lambda p:int(p.stem.split('-')[-1]))
        assert len(pages)>=5
        for start in range(0,len(pages),6):
            subset=pages[start:start+6];sheet=Image.new('RGB',(900,3*660),'#e7e9ed');draw=ImageDraw.Draw(sheet)
            for i,path in enumerate(subset):
                im=Image.open(path).convert('RGB');im.thumbnail((420,610))
                x=(i%2)*450+15;y=(i//2)*660+30;sheet.paste(im,(x,y))
                draw.text((x,y-20),'Page '+str(start+i+1),fill='black')
            sheet.save(render/f'contact-{start//6+1}.png')
        # Exact selected choices, predictions and code accompany the plotted summaries.
        for seed in [42,43,44]:
            for name in ['summary.json','selection.json','split-ids.json','fit-control.json','predictions.npz']:
                shutil.copyfile(ext/f'seed{seed}'/name,artifact/f'seed{seed}-{name}')
        for name in ['plan.json','source-sha256.json','status.json']:shutil.copyfile(ext/name,artifact/name)
        shutil.copyfile(root/'outputs/strengthening-preflight.json',artifact/'preflight.json')
        for name in ['code/activation_extension.py','code/test_activation_extension.py','code/common.py','code/coefficient_merge/core.py','code/merge_study/engine.py']:
            shutil.copyfile(root/name,artifact/('source__'+name.replace('/','__')))
        (artifact/'README.md').write_text('# Proposal-aligned activation and routing extension\n\nExploratory extension after v3 final inspection. The allocated L4 is used for all new FP32 controls; historical A100 baseline/timing stages stay distinct.\n\n- summary.json and CSVs: independently audited metrics and paired comparisons.\n- activation-controls.png, routing-controls.png, routing-impact.png: study figures.\n- pipeline/protocol PNG and SVG: presentation diagrams.\n- seed*-*: complete histories, locked choices, split IDs, shuffled permutations and prediction arrays.\n- plan, sources and preflight: exact execution evidence.\n- manifest.json: SHA256 of all other evidence files.\n\nOracle uses final GT and is an accuracy ceiling, not a deployment method. Source recognition is not expert utility.\n')
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(artifact.iterdir()) if p.is_file()}
        (artifact/'manifest.json').write_text(json.dumps({'files':files},indent=2))
        logs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'outputs').glob('course-*.log')}
        (report/'build-verification.json').write_text(json.dumps({'state':'compiled-and-rendered','pages':len(pages),
            'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'compiler':json.loads((root/'outputs/report-tools/ready.json').read_text()),
            'log_sha256':logs,'note':'Visual review of the rendered pages remains required before delivery.'},indent=2))
        mainzip=Path('/content/COMP9501-Part2-course-report-20261111.zip')
        include=[p for p in report.iterdir() if p.is_file() and p.suffix not in {'.aux','.out','.log','.synctex','.toc'}]
        include+=list(artifact.iterdir())
        with zipfile.ZipFile(mainzip,'w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(include):
                if path.is_file():z.write(path,path.relative_to(root).as_posix())
        reviewzip=Path('/content/COMP9501-Part2-report-review-20261111.zip')
        with zipfile.ZipFile(reviewzip,'w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(render.iterdir()):z.write(path,path.name)
            z.write(report/'report.pdf','report.pdf');z.write(textfile,'report-text.txt')
        metadata={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [mainzip,reviewzip]}
        Path('/content/COMP9501-course-materials-ready.json').write_text(json.dumps(metadata,indent=2))
        state.update(state='exported',pages=len(pages),archives=metadata);publish(None)
    except Exception as error:
        state.update(state='error',error=str(error),traceback=traceback.format_exc());publish('error');raise


if __name__=='__main__':main()
