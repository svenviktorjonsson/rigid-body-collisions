"""Exercise receipt integrity after malicious rehashing, without engine replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

BASE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('component_receipt_audit', BASE / 'audit.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ReceiptChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name) / 'archive'
        shutil.copytree(BASE, self.base)

    def receipt(self):
        return json.loads((self.base / 'twenty-two-replays.json').read_text())

    def save(self, data):
        (self.base / 'twenty-two-replays.json').write_text(json.dumps(data))

    def test_portable_without_current_host_runtime(self):
        real_digest = AUDIT.digest
        real_output = AUDIT.subprocess.check_output
        def digest(path):
            if str(path).startswith('/usr/') or str(path).endswith('/build/spatial/spatial_coulomb_replay'):
                raise AssertionError('portable audit read current runtime bytes')
            return real_digest(path)
        def output(command, *args, **kwargs):
            if command[0] == 'ldd':
                raise AssertionError('portable audit invoked current-host ldd')
            return real_output(command, *args, **kwargs)
        with patch.object(AUDIT, 'digest', digest), patch.object(AUDIT.subprocess, 'check_output', output):
            self.assertEqual(AUDIT.audit(self.base)['accepted_count'], 22)

    def test_updated_raw_response_does_not_bypass_physics(self):
        record = self.receipt(); first = record['records'][0]
        captured = json.loads((AUDIT.ROOT / first['capture']).read_text())
        k = next(i for i, dependency in enumerate(captured['dependencies']) if dependency < 0)
        # A small perturbation keeps the candidate dissipative so this fixture
        # specifically reaches the independent full-row projection gate.
        first['native']['p'][k] += .001
        first['native']['w'] = (np.asarray(captured['A']) @ first['native']['p'] - captured['b']).tolist()
        self.save(record)
        with self.assertRaisesRegex(AssertionError, 'Original full-row projection'):
            AUDIT.audit(self.base)

    def test_rehashed_committed_source_is_rejected(self):
        record = self.receipt(); path = 'spatial_backend/support_restart.h'
        with zipfile.ZipFile(self.base / 'execution-source.zip') as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        members[path] += b'\n// changed source\n'
        record['source_sha256'][path] = hashlib.sha256(members[path]).hexdigest()
        with zipfile.ZipFile(self.base / 'execution-source.zip', 'w') as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
        self.save(record)
        with self.assertRaisesRegex(AssertionError, 'differs from committed model'):
            AUDIT.audit(self.base)

    def test_rehashed_cost_cannot_exceed_declared_caps(self):
        record = self.receipt()
        record['records'][-1]['native']['stats']['support_svd_calls'] = 1025
        self.save(record)
        with self.assertRaisesRegex(AssertionError, 'Declared search budget exceeded'):
            AUDIT.audit(self.base)

    def test_missing_cost_counter_is_not_zero(self):
        record = self.receipt()
        del record['records'][0]['native']['stats']['support_svd_calls']
        self.save(record)
        with self.assertRaisesRegex(AssertionError, 'Required recorded search counter is missing'):
            AUDIT.audit(self.base)


if __name__ == '__main__':
    unittest.main()
