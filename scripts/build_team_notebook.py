"""Build a deterministic, credential-free Colab source snapshot."""
from pathlib import Path
import argparse,base64,hashlib,io,json,zipfile

ROOT=Path(__file__).resolve().parents[1]


def bundle_files():
    paths=[p for p in (ROOT/'code').rglob('*.py') if '__pycache__' not in p.parts]
    paths+=list((ROOT/'configs').glob('*.json'))
    paths+=[ROOT/'requirements.txt',ROOT/'scripts/colab_bundle.py',ROOT/'scripts/test_colab_bundle.py',ROOT/'assets/checkpoints-2026-10-25.json']
    paths+=[p for p in (ROOT/'part1-reproduction-20261025/code').glob('*.py')]
    paths+=[ROOT/'part1-reproduction-20261025/plan.json']
    paths+=[ROOT/'part1-reproduction-20261025/requirements.txt',
            ROOT/'part1-reproduction-20261025/presentation-20261028/plan.json']
    return {p.relative_to(ROOT).as_posix():p.read_bytes().replace(b'\r\n',b'\n') for p in sorted(set(paths))}


def build():
    files=bundle_files();manifest={'kind':'COMP9501 embedded source snapshot','files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}}
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,value in sorted({**files,'bundle-manifest.json':(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()}.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,28,0,0,0))
            info.create_system=3;info.external_attr=0o100644<<16
            info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,value)
    payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest();helper=files['scripts/colab_bundle.py']
    def md(text):return {'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
    def code(text):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':text.splitlines(keepends=True)}
    cells=[md('''# COMP9501 team runner

Open this notebook with your own authorized GitHub/Google account, or download the notebook from the private repository and upload it to Colab. The source snapshot is embedded; no GitHub token or runtime git clone is needed. GPU allocation and subscription benefits belong to each runtime account. See COLLABORATION.md in the repository.

Run all performs environment setup and offline checks. Training, checkpoint restoration and result export use the explicit switches below. Change the source in a Git branch/PR, then rebuild this launcher for a new snapshot. This runtime folder is an execution copy.
'''),code('''# Project source, checked before importing. Do not edit the encoded payload.
from pathlib import Path
import base64, hashlib, os, sys, subprocess, uuid
SOURCE_SNAPSHOT = DIGEST
_helper_bytes = base64.b64decode(HELPER)
assert hashlib.sha256(_helper_bytes).hexdigest() == HELPER_SHA
_bootstrap = {}
exec(compile(_helper_bytes, '<verified project bootstrap>', 'exec'), _bootstrap)
ROOT = Path('/content') / ('COMP9501-GP-' + SOURCE_SNAPSHOT[:12])
_bootstrap['materialize_source'](base64.b64decode(PAYLOAD), SOURCE_SNAPSHOT, ROOT)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
print('Project source verified:', SOURCE_SNAPSHOT)
print('Runtime folder:', ROOT)
'''.replace('HELPER_SHA',repr(hashlib.sha256(helper).hexdigest())).replace('HELPER',repr(base64.b64encode(helper).decode())).replace('DIGEST',repr(digest)).replace('PAYLOAD',repr(base64.b64encode(payload).decode()))),
code('''subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
'''),code('''checks = [('code', 'test_math.py'), ('code', 'test_protocol.py'), ('code', 'test_independent_protocol.py'), ('code', 'test_activation_extension.py'), ('code/functional_merge', 'test_maths.py'), ('code/coefficient_merge', 'test_core.py'), ('code/merge_study', 'test_engine.py'), ('part1-reproduction-20261025/code', 'test_csa*.py'), ('scripts', 'test_colab_bundle.py')]
for folder, pattern in checks:
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', folder, '-p', pattern], check=True)
'''),md('''## Optional: restore the existing 12 adapters

Get the checkpoint ZIP from the group's private file store. Upload it using Colab's Files panel or set the path to your own authorized Drive mount. GitHub collaborator access does not grant Drive access automatically. The archive remains outside Git.
'''),code('''CHECKPOINT_ARCHIVE = ''  # e.g. path to the uploaded ZIP; leave blank to skip
if CHECKPOINT_ARCHIVE:
    from colab_bundle import restore_checkpoints
    restore_checkpoints(CHECKPOINT_ARCHIVE, ROOT / 'assets/checkpoints-2026-10-25.json', ROOT)
    print('Checkpoint archive and every restored file verified.')
'''),md('''## Optional: train a fresh adapter bank

Select a GPU before enabling training. Choose a unique output directory. Each member keeps their own identity for code commits; MEMBER below is just the runtime record label. Existing source-calibration and exploratory-validation limits still apply.
'''),code('''MEMBER = 'member'  # use your own short alias
SEED = 42
RUN_TRAINING = False
RUN_NAME = MEMBER + '-bank-s' + str(SEED) + '-' + uuid.uuid4().hex[:8]
RUN_DIR = ROOT / 'outputs' / RUN_NAME
if RUN_TRAINING:
    assert not RUN_DIR.exists()
    subprocess.run([sys.executable, 'code/run_bank.py', '--config', 'configs/protocol-v2-a100.json', '--seed', str(SEED), '--output', str(RUN_DIR)], check=True)
'''),md('''## Optional: evaluate one restored adapter

This uses the original task-specific head and source validation protocol. Restore the checkpoint asset first. Public base model and dataset downloads are still required. Keep RUN_TRAINING disabled if you only want this evaluation.
'''),code('''RUN_EVALUATION = False
EVAL_TASK = 'sst2'  # sst2, mrpc, rte or cola
if RUN_EVALUATION:
    bank = ROOT / ('outputs/bank-v2-a100-s42' if SEED == 42 else 'outputs/replication-s43-s44/bank-s' + str(SEED))
    adapter = bank / EVAL_TASK / 'adapter'
    assert (adapter / 'adapter_model.safetensors').is_file(), 'Restore the checkpoint asset first.'
    RUN_NAME = MEMBER + '-eval-' + EVAL_TASK + '-s' + str(SEED) + '-' + uuid.uuid4().hex[:8]
    RUN_DIR = ROOT / 'outputs' / RUN_NAME
    RUN_DIR.mkdir(parents=True)
    subprocess.run([sys.executable, 'code/evaluate.py', '--config', 'configs/protocol-v2-a100.json', '--task', EVAL_TASK, '--adapter', str(adapter), '--output', str(RUN_DIR / 'result.json')], check=True)
'''),md('''## Optional: restore or train the independent v3 study

For the final Part 2 protocol, use the separate v3 checkpoint ZIP rather than the older bank above. Set exactly one recovery/training path. Restoring verifies the archive's file manifest and relocates partition paths; it does not retrain teachers. A fresh study trains all twelve teachers and evaluates the fixed weight-space grid, so it needs substantially more GPU time. The default switches run neither operation.
'''),code('''INDEPENDENT_CHECKPOINT_ARCHIVE = ''  # uploaded v3 checkpoint ZIP, or leave blank
RUN_FRESH_V3 = False
assert not (INDEPENDENT_CHECKPOINT_ARCHIVE and RUN_FRESH_V3), 'Choose restore or fresh training.'
V3_RUN_NAME = MEMBER + '-v3-' + uuid.uuid4().hex[:8]
V3_STUDY = ROOT / 'outputs' / V3_RUN_NAME
if INDEPENDENT_CHECKPOINT_ARCHIVE:
    subprocess.run([sys.executable, 'code/restore_independent_backup.py', '--archive', INDEPENDENT_CHECKPOINT_ARCHIVE, '--output', str(V3_STUDY)], check=True)
elif RUN_FRESH_V3:
    subprocess.run([sys.executable, 'code/run_independent_study.py', '--output', str(V3_STUDY)], check=True)
    subprocess.run([sys.executable, 'code/independent_baselines.py', '--study', str(V3_STUDY)], check=True)
if INDEPENDENT_CHECKPOINT_ARCHIVE or RUN_FRESH_V3:
    RUN_NAME, RUN_DIR = V3_RUN_NAME, V3_STUDY
'''),md('''## Optional: reproduce the direct activation and routing controls

Enable this only after the v3 study above is restored or completed. All five static profiles and both learned routers use the locked fit/selection/final roles. The final task partition was previously explored; this is an exploratory reproduction, not a new blind test. This command requires a CUDA GPU with at least 20 GiB VRAM (the recorded recovery run used L4). Select a new output directory and export it before the temporary VM disappears.
'''),code('''RUN_ACTIVATION_CONTROLS = False
if RUN_ACTIVATION_CONTROLS:
    assert (V3_STUDY / 'locked-config.json').exists(), 'Restore or train v3 first.'
    RUN_NAME = MEMBER + '-activation-' + uuid.uuid4().hex[:8]
    RUN_DIR = ROOT / 'outputs' / RUN_NAME
    subprocess.run([sys.executable, 'code/activation_extension.py', '--study', str(V3_STUDY), '--plan', 'configs/activation-strengthening-v1.json', '--output', str(RUN_DIR)], check=True)
'''),md('''## Export a completed run

Set RUN_DIR to the completed outputs subdirectory if needed. The ZIP includes a file manifest, member label, source snapshot and seed. Download it before the temporary VM disappears, or set EXPORT_DIRECTORY to a folder in your own authorized Drive mount. Nothing uploads automatically to the group's account.
'''),code('''EXPORT_RESULTS = False
EXPORT_DIRECTORY = ROOT / 'outputs' / 'exports'
DOWNLOAD_ZIP = True
if EXPORT_RESULTS:
    from colab_bundle import export_run
    is_v3 = bool(INDEPENDENT_CHECKPOINT_ARCHIVE or RUN_FRESH_V3 or RUN_ACTIVATION_CONTROLS)
    metadata = {'member': MEMBER, 'seed': None if is_v3 else SEED, 'training_seeds': [42, 43, 44] if is_v3 else [SEED], 'source_snapshot': SOURCE_SNAPSHOT, 'scope': 'Inspect exported manifests for the exact protocol and configuration.'}
    export = export_run(ROOT, RUN_DIR, EXPORT_DIRECTORY / (RUN_NAME + '.zip'), metadata)
    print('Export SHA-256:', export['sha256'])
    if DOWNLOAD_ZIP:
        from google.colab import files
        files.download(export['path'])
''')]
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'provenance':[]}},'cells':cells}
    record={'source_snapshot_sha256':digest,'zip_bytes':len(payload),'embedded_files':manifest['files'],'bootstrap_sha256':hashlib.sha256(helper).hexdigest(),'scope':'Credential-free source execution snapshot; no trained weights or raw datasets embedded. GPU/browser integration still requires each member account.'}
    return json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',json.dumps(record,indent=2)+'\n'


def chinese_notebook(notebook):
    value=json.loads(notebook)
    translations=json.loads((ROOT/'notebooks/team_notebook.zh-CN.json').read_text(encoding='utf-8'))
    markdown=[''.join(c['source']) for c in value['cells'] if c['cell_type']=='markdown']
    digest=hashlib.sha256(json.dumps(markdown,ensure_ascii=False).encode('utf-8')).hexdigest()
    if digest!=translations['source_markdown_sha256']:
        raise ValueError('English instructions changed: review the Chinese translation and source_markdown_sha256')
    headings={text.splitlines()[0] for text in markdown}
    if headings!=set(translations['sections']):raise ValueError('Chinese notebook section coverage mismatch')
    for cell in value['cells']:
        if cell['cell_type']=='markdown':
            heading=''.join(cell['source']).splitlines()[0]
            cell['source']=translations['sections'][heading].splitlines(keepends=True)
    return json.dumps(value,ensure_ascii=False,indent=2)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--print-markdown-digest',action='store_true');args=p.parse_args();notebook,record=build()
    if args.print_markdown_digest:
        markdown=[''.join(c['source']) for c in json.loads(notebook)['cells'] if c['cell_type']=='markdown']
        print(hashlib.sha256(json.dumps(markdown,ensure_ascii=False).encode('utf-8')).hexdigest());return
    targets={ROOT/'notebooks/team_colab.ipynb':notebook,ROOT/'notebooks/team_colab.zh-CN.ipynb':chinese_notebook(notebook),ROOT/'artifacts/team-launcher-2026-10-28.json':record}
    if args.check:
        if any(not p.exists() or p.read_text(encoding='utf-8')!=text for p,text in targets.items()):raise SystemExit('Team launcher is stale. Run python scripts/build_team_notebook.py')
        print('TEAM_LAUNCHER_CURRENT')
    else:
        for p,text in targets.items():p.write_bytes(text.encode())
        print('TEAM_LAUNCHER_BUILT',json.loads(record)['zip_bytes'])


if __name__=='__main__':main()
