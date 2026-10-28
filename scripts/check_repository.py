"""Optional syntax, links, privacy and executable-notebook checks."""
from pathlib import Path
import ast,json,re,sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
IGNORED={'.git','.tools','.venv','venv','__pycache__','.pytest_cache','.cache','outputs','data','datasets','.ipynb_checkpoints'}
TOKEN=re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})\b')
LOCAL_PATH=re.compile(r'(?i)(?:[a-z]:[\\/](?:users|dsh-campus)[\\/]|[/]Users[/][^/]+[/]|[/]home[/][^/]+[/])')


def main():
    files=[p for p in ROOT.rglob('*') if p.is_file() and not (set(p.relative_to(ROOT).parts)&IGNORED)]
    errors=[];warnings=[];counts={'python':0,'json':0,'notebooks':0,'markdown':0}
    for p in files:
        relative=p.relative_to(ROOT).as_posix()
        if p.suffix in {'.pt','.pth','.safetensors','.npz','.npy','.pkl','.zip'}:
            warnings.append(relative+': consider sharing large model/data files separately')
        if p.suffix not in {'.py','.md','.json','.ipynb','.txt','.yml','.yaml'}:continue
        try:text=p.read_text(encoding='utf-8-sig')
        except UnicodeError:
            errors.append(relative+': invalid UTF-8');continue
        if TOKEN.search(text):errors.append(relative+': possible credential detected (value withheld)')
        if LOCAL_PATH.search(text):errors.append(relative+': private local absolute path detected')
        try:
            if p.suffix=='.py':ast.parse(text,filename=relative);counts['python']+=1
            if p.suffix=='.json':json.loads(text);counts['json']+=1
            if p.suffix=='.ipynb':
                notebook=json.loads(text);assert notebook['nbformat']==4
                for c in notebook['cells']:
                    assert c['cell_type'] in {'markdown','code','raw'}
                    if c['cell_type']=='code':
                        source=''.join(c['source'])
                        if not any(line.lstrip().startswith(('!','%','?')) for line in source.splitlines()):ast.parse(source)
                counts['notebooks']+=1
            if p.suffix=='.md':
                counts['markdown']+=1
                for ref in re.findall(r'\]\(([^)]+)\)',text):
                    if '://' in ref or ref.startswith(('#','mailto:')):continue
                    dest=(p.parent/unquote(ref.split('#',1)[0])).resolve()
                    if not dest.is_relative_to(ROOT):errors.append(relative+': link leaves the checkout: '+ref)
                    elif not dest.exists():errors.append(relative+': missing relative target: '+ref)
        except (SyntaxError,ValueError,KeyError,AssertionError) as error:errors.append(relative+': '+str(error))
    if (ROOT/'notebooks/team_colab.ipynb').exists():
        try:
            from build_team_notebook import build,chinese_notebook
            notebook,record=build()
            if (ROOT/'notebooks/team_colab.ipynb').read_text(encoding='utf-8')!=notebook or (ROOT/'artifacts/team-launcher-2026-10-28.json').read_text(encoding='utf-8')!=record:
                errors.append('Colab source snapshot is stale; run python scripts/build_team_notebook.py')
            chinese_path=ROOT/'notebooks/team_colab.zh-CN.ipynb'
            if chinese_path.exists() and chinese_path.read_text(encoding='utf-8')!=chinese_notebook(notebook):
                errors.append('Chinese Colab source snapshot is stale; run python scripts/build_team_notebook.py')
        except (ValueError,FileNotFoundError) as error:errors.append(str(error))
    print(json.dumps({'checked_files':len(files),'counts':counts,'warnings':warnings,'errors':errors},ensure_ascii=False,indent=2))
    return bool(errors)


if __name__=='__main__':sys.exit(main())
