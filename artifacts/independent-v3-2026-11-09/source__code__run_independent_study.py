"""Cloud-only fixed three-seed teacher/coefficient experiment, with durable logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    import torch
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('This study requires the authorized Colab A100 runtime')
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = {'state': 'running', 'jobs': [], 'started_epoch': time.time()}
    def publish():
        tmp = out/'status.tmp'
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, out/'status.json')
    def run(name, script, *values):
        status['current'] = name
        publish()
        start = time.perf_counter()
        with (out/(name+'.log')).open('w') as log:
            completed = subprocess.run([sys.executable, str(root/'code'/script), *map(str,values)],
                                       cwd=root, stdout=log, stderr=subprocess.STDOUT)
        status['jobs'].append({'name': name, 'returncode': completed.returncode,
                               'seconds': time.perf_counter()-start})
        publish()
        if completed.returncode:
            raise RuntimeError('Inspect '+name+'.log')
    try:
        cfg = json.loads((root/'configs/independent-v3.json').read_text())
        plan = root/'configs/independent-coefficients-v3.json'
        (out/'coefficient-plan.json').write_bytes(plan.read_bytes())
        sources = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (root/'code').rglob('*.py') if '__pycache__' not in p.parts}
        (out/'source-sha256.json').write_text(json.dumps(sources, indent=2))
        manifest = out/'partitions.json'
        run('partitions', 'independent_protocol.py', '--config', root/'configs/independent-v3.json', '--output', manifest)
        cfg.update(partition_manifest=str(manifest), partition_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   dataset_revision=json.loads(manifest.read_text())['dataset_revision'])
        config = out/'locked-config.json'
        config.write_text(json.dumps(cfg, indent=2))
        for seed in cfg['seeds']:
            bank = out/f'bank-s{seed}'
            bank.mkdir()
            (bank/'status.json').write_text(json.dumps({'seed': seed, 'state': 'training', 'config': cfg}))
            for task in cfg['tasks']:
                run(f'train-s{seed}-{task}', 'train_lora.py', '--config', config, '--task', task,
                    '--seed', seed, '--output', bank/task)
            (bank/'status.json').write_text(json.dumps({'seed': seed, 'state': 'passed', 'config': cfg}))
            run(f'coefficients-s{seed}', 'coefficient_merge/run.py', '--bank', bank,
                '--config', config, '--plan', out/'coefficient-plan.json', '--output', out/f'coefficients-s{seed}')
        status.update(state='passed', current=None, finished_epoch=time.time())
        publish()
    except Exception as error:
        status.update(state='error', error=str(error), traceback=traceback.format_exc())
        publish()
        raise


if __name__ == '__main__':
    main()
