import copy
import unittest
from independent_protocol import ROLES, content_key, partition_rows, validate_manifest


class PartitionChecks(unittest.TestCase):
    def setUp(self):
        self.rows = [{'idx': i, 'text': f'Example {i}', 'label': i % 2} for i in range(3000)]

    def test_duplicates_and_labels_do_not_cross_roles(self):
        duplicate = {'idx': 9000, 'text': '  EXAMPLE   0 ', 'label': 1}
        parts = partition_rows(self.rows + [duplicate], ('text',))
        reverse_labels = [{**r, 'label': 1-r['label']} for r in self.rows]
        self.assertEqual(parts, partition_rows(reverse_labels, ('text',)))
        self.assertEqual(content_key(['Ａbc  def']), content_key(['abc def']))
        manifest = {'protocol': 'content-grouped-v3', 'tasks': {'a': {'partitions': parts}}}
        validate_manifest(manifest)
        all_ids = [i for role in ROLES for i in parts[role]['sample_ids']]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertNotIn(9000, all_ids)

    def test_cross_task_content_leak_is_rejected(self):
        parts = partition_rows(self.rows, ('text',))
        other = copy.deepcopy(parts)
        other['calibration']['content_sha256'][0] = parts['train']['content_sha256'][0]
        manifest = {'protocol': 'content-grouped-v3', 'tasks': {
            'a': {'partitions': parts}, 'b': {'partitions': other}}}
        with self.assertRaisesRegex(ValueError, 'Content crosses'):
            validate_manifest(manifest)


if __name__ == '__main__':
    unittest.main()
