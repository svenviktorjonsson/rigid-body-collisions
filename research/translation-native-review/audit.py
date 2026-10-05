"""Independent checks of archived support solves; never reruns production."""
from pathlib import Path
import json,hashlib,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.coulomb_diagnostics import System
BASE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def audit(base=BASE):
 base=Path(base);manifest=json.loads((base/'manifest.json').read_text())
 for relative,expected in manifest['files'].items():
  if sha(base/relative)!=expected:raise AssertionError('Archived artifact changed: '+relative)
 for relative in ['plan.json','v2/plan.json','v3/plan.json']:
  plan=json.loads((base/relative).read_text())
  for name,expected in plan['source_hashes'].items():
   source=ROOT/name
   try:source=base/source.relative_to(BASE)
   except ValueError:pass
   if sha(source)!=expected:raise AssertionError('Frozen numerical source changed: '+name)
  for name,expected in plan.get('captures',{plan['capture']:plan['capture_sha256']}).items():
   if sha(ROOT/name)!=expected:raise AssertionError('Captured physical system changed: '+name)
 extension=json.loads((base/'v3/combined22-plan.json').read_text())
 if sha(base/'v3/combined_replay.cpp')!=extension['combined_driver_sha256'] or sha(base/'v3/checks.cpp')!=extension['control_source_sha256']:raise AssertionError('Frozen validation driver changed')
 primary=ROOT/'research/hull-translation-completion/results/execution-source.zip'
 if sha(primary)!=extension['production_source_zip_sha256']:raise AssertionError('Frozen primary source archive changed')
 with zipfile.ZipFile(primary) as archive:
  for name in archive.namelist():
   if name.startswith('spatial_backend/') and name.endswith('.h') and archive.read(name)!=(base/'production52'/Path(name).name).read_bytes():raise AssertionError('Copied primary numerical source changed')
 rows=[json.loads(line) for line in (base/'v3/combined22-native.jsonl').read_text().splitlines()]
 corpus=json.loads((base/'v3/combined22-plan.json').read_text())['corpus']
 if len(rows)!=22 or [r['capture'] for r in rows]!=corpus:raise AssertionError('Original22 capture corpus incomplete or reordered')
 for receipt in rows:
  d=json.loads((ROOT/receipt['capture']).read_text());gate=System.from_dump(d).gate(np.array(receipt['p']),d['tolerance_m_s'])
  if not receipt['accepted'] or not gate['accepted']:raise AssertionError('Original physical gate fails: '+receipt['capture'])
  if receipt['svd_calls']>1024 or receipt['helper_calls']>8 or receipt['passes']>8 or receipt['largest_reduced_rows']>64:raise AssertionError('V3 search cap exceeded')
  if receipt['pressure_svd_calls']>1024 or receipt['pivot_attempts']>8:raise AssertionError('Separately declared guide cap exceeded')
 for version in ['', 'v2/', 'v3/']:
  name='native-first261.jsonl' if not version else version+'native261-243.jsonl'
  results=[json.loads(line) for line in (base/name).read_text().splitlines()]
  for r in results:
   d=json.loads((ROOT/r['capture']).read_text());g=System.from_dump(d).gate(np.array(r['p']),d['tolerance_m_s'])
   if not r['accepted'] or not g['accepted']:raise AssertionError('Advertised standalone physical solve fails')
 failed=json.loads((base/'native-first243.jsonl').read_text());d=json.loads((ROOT/failed['capture']).read_text())
 if failed['accepted'] or failed['svd_calls']!=0 or failed['largest_reduced_rows']<=64 or failed['p']!=d['p']:raise AssertionError('Retained V1 structural decline changed')
 return dict(original_captures=22,accepted=22,standalone_successes=5,retained_structural_declines=1)
if __name__=='__main__':print(json.dumps(audit(),indent=2))
