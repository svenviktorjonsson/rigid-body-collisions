"""Rehashed false qualification and physically false traces must still reject."""
import contextlib,hashlib,io,json,shutil,tempfile,unittest,zipfile
from pathlib import Path
from research.audit_shared_shake import D,audit

class SharedShakingAudit(unittest.TestCase):
    def test_authentic_archive(self):
        with contextlib.redirect_stdout(io.StringIO()):self.assertTrue(audit()['qualified'])

    def mutate_trace(self,mutator):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'study';shutil.copytree(D,d);p=d/'results/traces.zip'
            with zipfile.ZipFile(p) as z:traces={n:z.read(n) for n in z.namelist()}
            n='candidate_0.json';r=json.loads(traces[n]);mutator(r);traces[n]=json.dumps(r).encode()
            with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
                for n,c in traces.items():z.writestr(n,c)
            q=d/'results/summary.json';s=json.loads(q.read_text());s['hashes']['traces.zip']=hashlib.sha256(p.read_bytes()).hexdigest();q.write_text(json.dumps(s))
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(AssertionError):audit(d)

    def test_rehashed_wrong_sphere_mass(self):
        self.mutate_trace(lambda r:r['mass'].__setitem__(1,2.))

    def test_rehashed_wrong_prescribed_wall(self):
        self.mutate_trace(lambda r:r['states'][1][0].__setitem__(0,r['states'][1][0][0]+.001))

    def test_false_reference_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'study';shutil.copytree(D,d);p=d/'results/summary.json';s=json.loads(p.read_text());s['reference_qualified']=False;p.write_text(json.dumps(s))
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(AssertionError):audit(d)
