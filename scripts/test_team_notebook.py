"""Regression for differing Windows/Linux ZIP creator defaults in source snapshots."""
import unittest
from unittest.mock import patch
import zipfile
import build_team_notebook as builder

REAL_ZIPINFO = zipfile.ZipInfo


class SnapshotChecks(unittest.TestCase):
    def test_windows_and_unix_creator_defaults_produce_same_snapshot(self):
        outputs = []
        for creator in (0, 3):
            class HostZipInfo(REAL_ZIPINFO):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.create_system = creator
            with patch.object(builder.zipfile, 'ZipInfo', HostZipInfo):
                outputs.append(builder.build())
        self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
