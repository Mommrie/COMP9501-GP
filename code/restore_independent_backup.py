"""Verify and restore the v3 checkpoint export into a new directory."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def restore(archive, output):
    archive, output = Path(archive), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC mismatch')
        manifest = json.loads(z.read('backup-manifest.json'))['files']
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(manifest) | {'backup-manifest.json'}:
            raise ValueError('Unlisted or repeated archive members')
        for name, digest in manifest.items():
            path = PurePosixPath(name.replace('\\', '/'))
            if path.is_absolute() or '..' in path.parts or ':' in name:
                raise ValueError('Unsafe archive path')
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError('File checksum mismatch: '+name)
        output.mkdir(parents=True, exist_ok=False)
        for name in z.namelist():
            target = (output/name).resolve()
            if not target.is_relative_to(output):
                raise ValueError('Archive member leaves target')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
    evidence = output/'evidence'
    cfg = json.loads((evidence/'locked-config.json').read_text())
    for name in ['partitions.json','coefficient-plan.json','source-sha256.json','status.json']:
        shutil.copyfile(evidence/name, output/name)
    cfg['partition_manifest'] = str(output/'partitions.json')
    assert hashlib.sha256((output/'partitions.json').read_bytes()).hexdigest()==cfg['partition_manifest_sha256']
    (output/'locked-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for seed in cfg['seeds']:
        bank = output/f'bank-s{seed}'
        for task in cfg['tasks']:
            for name in ['run.json','data-manifest.json']:
                shutil.copyfile(evidence/f'bank-s{seed}__{task}__{name}', bank/task/name)
        (bank/'status.json').write_text(json.dumps({'seed':seed,'state':'passed','config':cfg,
                  'provenance':'Restored completed teacher bank; original run JSON and archive preserved.'},indent=2)+'\n')
        coeff = output/f'coefficients-s{seed}'
        coeff.mkdir(exist_ok=True)
        for name in ['summary.json','selection.json','split-ids.json']:
            shutil.copyfile(evidence/f'coefficients-s{seed}__{name}',coeff/name)
    baseline = output/'weight-baselines'
    baseline.mkdir()
    for name in ['summary.json','status.json']:
        shutil.copyfile(evidence/'weight-baselines-summary.json',baseline/name)
    for seed in cfg['seeds']:
        shutil.copyfile(evidence/f'weight-baselines-selection-s{seed}.json',baseline/f'selection-s{seed}.json')
    result={'state':'passed','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'restore_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'verified_files':len(manifest),'adapter_count':len(cfg['seeds'])*len(cfg['tasks']),
            'partition_sha256':cfg['partition_manifest_sha256'],
            'scope':'Weights and original experiment evidence restored; base/tokenizer downloaded separately at pinned Hub revision.'}
    (output/'restore-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    print(json.dumps(restore(args.archive,args.output)))


if __name__=='__main__':
    main()
