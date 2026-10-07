"""Independent archived physics/edge/observer checks, retaining real failures."""
import hashlib,json,subprocess
from pathlib import Path
from importlib import import_module
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];load=lambda p:json.loads(p.read_text());sha=lambda raw:hashlib.sha256(raw).hexdigest()
independent=import_module('research.rapid-friction.audit');base=import_module('research.rapid-friction.run');gates=load(ROOT/'research/rapid-friction/plan.json');records=[]
for folder in ['world-results']:
 D=H/folder;provenance=load(D/'provenance.json');source=provenance['source'];live=True
 for path,digest in dict(provenance['guards'],**provenance.get('runtime',{})).items():
  p=Path(path);live &= sha(p.read_bytes())==digest
  if p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'build'):
   raw=subprocess.check_output(['git','show',source+':'+str(p.relative_to(ROOT))],cwd=ROOT)
  else:raw=p.read_bytes()
  assert sha(raw)==digest,path
 scenes=load(D/'scenes.json');summary=load(D/'summary.json');worlds=[]
 for name,entry in scenes.items():
  refs=[];edges=[]
  for p in sorted((D/name).glob('reference_*.json')):
   r=load(p);result=r['result'];assert np.isfinite(result['states']).all()
   physical=independent.physical(entry,result);assert physical==r['physical']['passed']
   assert abs(result['times'][-1]-.12)<1e-12
   refs.append(r)
   if len(refs)>1:
    a,b=refs[-2:];errors=base.errors(2,a['result'],b['result']);passed=independent.physical(entry,a['result']) and physical and all(errors[k]<=v/4 for k,v in gates['trajectory_budgets']['2'].items());edges.append({'passed':bool(passed),'errors':errors})
  if name in summary:
   stored=summary[name];assert stored['full_histories']==len(refs);assert [e['passed'] for e in edges]==[e['passed'] for e in stored['edges']]
   for a,b in zip(edges,stored['edges']):assert a['errors']==b['errors']
  if refs:worlds.append({'case':name,'full_histories':len(refs),'physical_passes':sum(independent.physical(entry,r['result']) for r in refs),'edges':edges,'reference_qualified':len(edges)>=2 and all(e['passed'] for e in edges[-2:])})
 records.append({'archive':folder,'historical_guards_verified':True,'current_live_unchanged':bool(live),'worlds':worlds,'procedural_failure_retained':(D/'failure.log').exists()})
receipt={'passed':True,'archives':records,'scope':'Independent unchanged physical gates, full trajectories, frozen provenance and every actual failed refinement edge. Analytical controls archived separately; no trajectory or performance qualification.'}
(H/'independent-archive-audit.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print('Joint planar independent archive audit PASS')
