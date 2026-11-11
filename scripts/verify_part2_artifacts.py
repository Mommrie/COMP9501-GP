"""Check the published Part 2 evidence and final PDF without importing ML packages."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def verify_manifest(folder):
    files = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))['files']
    for name, expected in files.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError('Missing or unsafe evidence path: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Evidence checksum mismatch: ' + str(path.relative_to(ROOT)))
    return len(files)


def main():
    counts = {}
    for name in ['independent-v3-2026-11-09', 'part2-followup-2026-11-09',
                 'activation-strengthening-2026-11-11']:
        folder = ROOT / 'artifacts' / name
        counts[name] = verify_manifest(folder)
    report = ROOT / 'report-20261109'
    info = json.loads((report / 'build-verification.json').read_text(encoding='utf-8'))
    actual = hashlib.sha256((report / 'report.pdf').read_bytes()).hexdigest()
    if actual != info['pdf_sha256']:
        raise ValueError('Compiled report PDF checksum mismatch')
    review = json.loads((report / 'visual-review.json').read_text(encoding='utf-8'))
    if review['pdf_sha256'] != actual or review['state'] != 'passed':
        raise ValueError('Final PDF has not passed recorded visual review')
    print(json.dumps({'state': 'passed', 'verified_evidence_files': counts,
                      'pdf_pages': info['pages'], 'pdf_sha256': actual}, indent=2))


if __name__ == '__main__':
    main()
