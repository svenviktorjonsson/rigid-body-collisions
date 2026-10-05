"""Frozen numeric/source claims remain checked after consistent rehashing."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
from research.audit_predictive_contact_review import DEFAULT,audit


class PredictiveContactArchiveAudit(unittest.TestCase):
    def rewritten(self,filename,mutate):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        directory=Path(temporary.name)/'archive';shutil.copytree(DEFAULT,directory)
        path=directory/filename
        if filename.endswith('.json'):
            data=json.loads(path.read_text());mutate(data);path.write_text(json.dumps(data))
        else:
            with zipfile.ZipFile(path) as z:data={n:z.read(n) for n in z.namelist()}
            mutate(data)
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
                for name,content in data.items():z.writestr(name,content)
        manifest=directory/'artifact-hashes.json';hashes=json.loads(manifest.read_text())
        hashes[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(hashes))
        return directory

    def test_actual_archive_passes_without_native_executable(self):
        result=audit();self.assertEqual(result['native_cases'],42);self.assertEqual(result['discovery_cases'],30)
        self.assertLess(result['maximum_endpoint_identity_error'],1e-14)

    def test_rehashed_false_angular_conservation_claim_is_rejected(self):
        def change(rows):
            row=next(r for r in rows if r['gap_m']==.001 and r['pair_mu']==1 and not r['common_point'])
            row['delta_angular_momentum_kg_m2_s']=[0,0,0]
        directory=self.rewritten('native-harness.json',change)
        with self.assertRaisesRegex(ValueError,'native endpoint torque'):audit(directory)

    def test_rehashed_false_energy_ledger_is_rejected(self):
        def change(rows):rows[0]['energy_J'][-1]=-1.
        directory=self.rewritten('full-engine.json',change)
        with self.assertRaisesRegex(ValueError,'full energy'):audit(directory)

    def test_rehashed_changed_frozen_solver_source_is_rejected(self):
        def change(files):files['spatial_backend/coulomb.h']+=b'\n// rewritten after the experiment\n'
        directory=self.rewritten('execution-source.zip',change)
        with self.assertRaisesRegex(ValueError,'Frozen diagnostic source changed'):audit(directory)
