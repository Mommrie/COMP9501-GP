import hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from colab_bundle import materialize_source,restore_checkpoints,export_run


def source_zip(files):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:
        for n,data in files.items():z.writestr(n,data)
        z.writestr('bundle-manifest.json',json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}}))
    value=buffer.getvalue();return value,hashlib.sha256(value).hexdigest()


class Tests(unittest.TestCase):
    def test_idempotent_import_and_changed_workspace_protection(self):
        payload,digest=source_zip({'code/example.py':b'print(1)\n'})
        with tempfile.TemporaryDirectory() as temp:
            root=materialize_source(payload,digest,temp);materialize_source(payload,digest,temp)
            (root/'code/example.py').write_text('local edit')
            with self.assertRaises(FileExistsError):materialize_source(payload,digest,temp)
            self.assertEqual((root/'code/example.py').read_text(),'local edit')

    def test_corruption_and_traversal_create_no_files(self):
        for name in ['../escape.txt','safe'+chr(92)+'escape.txt','safe//duplicate.txt']:
            payload,digest=source_zip({name:b'bad'})
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):materialize_source(payload,digest,temp)
                self.assertEqual(list(Path(temp).iterdir()),[])
                with self.assertRaises(ValueError):materialize_source(payload,'0'*64,temp)

    def test_checkpoint_requires_exact_member_hashes_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=root/'weights.zip'
            with zipfile.ZipFile(a,'w') as z:z.writestr('outputs/bank/model.txt','weights')
            manifest={'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'files':{'outputs/bank/model.txt':'0'*64}}
            m=root/'manifest.json';m.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):restore_checkpoints(a,m,root/'restored')
            self.assertFalse((root/'restored').exists())

    def test_export_is_unique_and_records_source_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'outputs/run-1';run.mkdir(parents=True);(run/'result.json').write_text('{}')
            destination=root/'exports/result.zip';metadata={'member':'member','source_snapshot':'abc','seed':42}
            result=export_run(root,run,destination,metadata)
            self.assertEqual(result['sha256'],hashlib.sha256(destination.read_bytes()).hexdigest())
            with zipfile.ZipFile(destination) as z:self.assertEqual(json.loads(z.read('run-metadata.json'))['source_snapshot'],'abc')
            with self.assertRaises(FileExistsError):export_run(root,run,destination,metadata)


if __name__=='__main__':unittest.main()
