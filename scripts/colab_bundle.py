"""Credential-free source import, checked checkpoint restore and run export."""
from pathlib import Path,PurePosixPath
import hashlib,io,json,os,stat,zipfile


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _entries(archive,max_bytes):
    entries=archive.infolist();names=[i.filename for i in entries]
    if len(entries)>5000 or len(names)!=len(set(names)):raise ValueError('Duplicate or excessive ZIP entries')
    if sum(i.file_size for i in entries)>max_bytes:raise ValueError('ZIP exceeds uncompressed-size limit')
    for item in entries:
        name=item.filename;path=PurePosixPath(name)
        if chr(92) in name:raise ValueError('Backslash in ZIP member')
        if chr(92) in name:raise ValueError('Backslash in ZIP member')
        if item.is_dir() or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or not path.parts or path.as_posix()!=name:
            raise ValueError('Invalid archive member')
        if stat.S_ISLNK(item.external_attr>>16):raise ValueError('Symlink archive member')
    return entries


def _restore(archive,files,destination,max_bytes):
    destination=Path(destination).resolve();entries=_entries(archive,max_bytes)
    if {e.filename for e in entries}!=set(files):raise ValueError('Archive file list does not match manifest')
    # Validate everything before creating a file, including collision checks.
    for item in entries:
        content=archive.read(item);expected=files[item.filename]
        if hashlib.sha256(content).hexdigest()!=expected:raise ValueError('Archive member checksum mismatch')
        target=destination/item.filename
        if not target.resolve().is_relative_to(destination):raise ValueError('Path escapes destination')
        if target.exists() and (not target.is_file() or sha256_file(target)!=expected):
            raise FileExistsError('Existing content differs: '+item.filename)
    for item in entries:
        target=destination/item.filename
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(item))
    return destination


def materialize_source(payload,expected_sha,destination):
    if hashlib.sha256(payload).hexdigest()!=expected_sha:raise ValueError('Source bundle checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        _entries(archive,64*1024*1024)
        manifest=json.loads(archive.read('bundle-manifest.json'))
        files=manifest['files'];files['bundle-manifest.json']=hashlib.sha256(archive.read('bundle-manifest.json')).hexdigest()
        return _restore(archive,files,destination,64*1024*1024)


def restore_checkpoints(archive_path,manifest_path,destination):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if sha256_file(archive_path)!=manifest['sha256']:raise ValueError('Checkpoint ZIP checksum mismatch')
    if not all(name.startswith('outputs/') for name in manifest['files']):raise ValueError('Checkpoint manifest must contain outputs only')
    with zipfile.ZipFile(archive_path) as archive:
        return _restore(archive,manifest['files'],destination,512*1024*1024)


def export_run(project_root,run_directory,destination,metadata):
    root=Path(project_root).resolve();run=Path(run_directory).resolve();dest=Path(destination).resolve()
    if not run.is_relative_to(root/'outputs') or not run.is_dir():raise ValueError('Choose an existing outputs run directory')
    if dest.is_relative_to(run):raise ValueError('Export ZIP must be outside the selected run')
    files=sorted(p for p in run.rglob('*') if p.is_file())
    if not files:raise ValueError('Run directory is empty')
    if any(not p.resolve().is_relative_to(run) for p in files):raise ValueError('Symlink leaves run directory')
    record={**metadata,'files':{p.relative_to(root).as_posix():sha256_file(p) for p in files}}
    dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p,p.relative_to(root).as_posix())
        archive.writestr('run-metadata.json',json.dumps(record,indent=2)+'\n')
    return {'path':str(dest),'sha256':sha256_file(dest),'bytes':dest.stat().st_size}
