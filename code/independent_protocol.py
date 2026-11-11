"""Content-grouped, label-blind partitions for the next experimental round."""
import hashlib
import json
import unicodedata

ROLES = ('train', 'calibration', 'coefficient_fit', 'coefficient_selection', 'validation')
THRESHOLDS = (.65, .73, .81, .89, 1.)
LIMITS = (4096, 128, 128, 128, 256)


def content_key(texts):
    normalized = [' '.join(unicodedata.normalize('NFKC', s).casefold().split()) for s in texts]
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False).encode()).hexdigest()


def partition_rows(rows, fields, seed=20261109):
    """Identical field tuples receive the same role, including across tasks.

    Labels are never consulted. Keep one representative per normalized tuple;
    order/caps depend only on content hashes. Unselected rows remain unused.
    """
    groups = {}
    for row in rows:
        key = content_key([row[f] for f in fields])
        groups.setdefault(key, int(row['idx']))
    candidates = {role: [] for role in ROLES}
    for key, idx in groups.items():
        digest = hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()
        value = int(digest[:16], 16) / 2**64
        role = next(r for r, bound in zip(ROLES, THRESHOLDS) if value < bound)
        candidates[role].append((digest, idx, key))
    return {role: {'sample_ids': [v[1] for v in sorted(candidates[role])[:cap]],
                   'content_sha256': [v[2] for v in sorted(candidates[role])[:cap]]}
            for role, cap in zip(ROLES, LIMITS)}


def validate_manifest(manifest):
    if manifest['protocol'] != 'content-grouped-v3':
        raise ValueError('Unknown independent protocol')
    global_roles = {}
    for task, record in manifest['tasks'].items():
        seen = set()
        for role in ROLES:
            part = record['partitions'][role]
            ids, keys = part['sample_ids'], part['content_sha256']
            if not ids or len(ids) != len(set(ids)) or len(ids) != len(keys) or seen.intersection(ids):
                raise ValueError(f'Invalid or overlapping IDs: {task}/{role}')
            seen.update(ids)
            for key in keys:
                prior = global_roles.setdefault(key, role)
                if prior != role:
                    raise ValueError('Content crosses information boundaries')
    return manifest


def main():
    import argparse
    from pathlib import Path
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from common import FIELDS, read_config, write_json
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    cfg = read_config(args.config)
    revision = cfg.get('dataset_revision') or HfApi().dataset_info('nyu-mll/glue').sha
    manifest = {'protocol': 'content-grouped-v3', 'split_seed': 20261109,
                'dataset_revision': revision, 'source_split': 'train', 'tasks': {},
                'scope': 'Fresh teachers and internally disjoint partitions; previously studied datasets, not a pristine external test.'}
    for task in cfg['tasks']:
        raw = load_dataset('nyu-mll/glue', task, split='train', revision=revision)
        manifest['tasks'][task] = {'fingerprint': raw._fingerprint,
                                  'partitions': partition_rows(raw, FIELDS[task])}
    validate_manifest(manifest)
    write_json(out, manifest)
    print(json.dumps({t: {r: len(v['sample_ids']) for r, v in d['partitions'].items()}
                      for t, d in manifest['tasks'].items()}))


if __name__ == '__main__':
    main()
