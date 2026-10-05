"""Evidence-integrity regressions: reported gates cannot replace recomputation."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
from research.audit_fast_shake_diagnostic import DIRECTORY,audit

class FastShakeAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.directory=Path(self.temporary.name)/'evidence'
        shutil.copytree(DIRECTORY,self.directory)

    def tearDown(self):self.temporary.cleanup()

    def reseal(self):
        path=self.directory/'artifact-hashes.json';manifest=json.loads(path.read_text())
        for name in manifest:manifest[name]=hashlib.sha256((self.directory/name).read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest))

    def test_real_archive_independently_qualifies_both_references(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit(self.directory,check_git=False),dict(fixed=True,travel=True))

    def test_consistently_rehashed_forged_qualification_is_rejected(self):
        path=self.directory/'corrected-extension-summary.json';summary=json.loads(path.read_text())
        summary['fixed']['qualified']=False;path.write_text(json.dumps(summary));self.reseal()
        with self.assertRaisesRegex(AssertionError,'Qualification fixed'):audit(self.directory,check_git=False)

    def test_consistently_rehashed_changed_initial_geometry_is_rejected(self):
        name='fixed_5us.json';path=self.directory/name;trial=json.loads(path.read_text())
        trial['states'][0][1][0]+=.01;path.write_text(json.dumps(trial))
        for archive_name in ('traces.zip','extension-traces.zip','corrected-extension-traces.zip'):
            path=self.directory/archive_name
            with zipfile.ZipFile(path) as archive:members={n:archive.read(n) for n in archive.namelist()}
            if name in members:members[name]=json.dumps(trial).encode()
            with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
                for key,value in members.items():archive.writestr(key,value)
        self.reseal()
        with self.assertRaisesRegex(AssertionError,'Initial position'):audit(self.directory,check_git=False)

if __name__=='__main__':unittest.main()
