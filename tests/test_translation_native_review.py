"""Archived contact evidence must remain verifiable even after receipt rehashing."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('translation_contact_audit',ROOT/'research/translation-native-review/audit.py')
AUDIT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(AUDIT)
class TranslationEvidenceTests(unittest.TestCase):
 def setUp(self):
  self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup);self.base=Path(self.temporary.name)/'archive';shutil.copytree(AUDIT.BASE,self.base)
 def rehash(self,relative):
  p=self.base/'manifest.json';d=json.loads(p.read_text());d['files'][relative]=hashlib.sha256((self.base/relative).read_bytes()).hexdigest();p.write_text(json.dumps(d))
 def test_original_archive(self):
  self.assertEqual(AUDIT.audit(self.base)['accepted'],22)
 def test_rehashed_impulse_still_requires_original_law(self):
  p=self.base/'v3/combined22-native.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()];d=json.loads((ROOT/rows[0]['capture']).read_text());k=next(i for i,x in enumerate(d['dependencies']) if x<0);rows[0]['p'][k]+=1;p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n');self.rehash('v3/combined22-native.jsonl')
  with self.assertRaisesRegex(AssertionError,'Original physical gate'):AUDIT.audit(self.base)
 def test_rehashed_source_still_requires_frozen_plan(self):
  p=self.base/'v3/support_restart.h';p.write_text(p.read_text()+'\n// altered archive\n');self.rehash('v3/support_restart.h')
  with self.assertRaisesRegex(AssertionError,'Frozen numerical source'):AUDIT.audit(self.base)
 def test_rehashed_cost_still_requires_declared_caps(self):
  p=self.base/'v3/combined22-native.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()];rows[0]['svd_calls']=1025;p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n');self.rehash('v3/combined22-native.jsonl')
  with self.assertRaisesRegex(AssertionError,'V3 search cap'):AUDIT.audit(self.base)
if __name__=='__main__':unittest.main()
