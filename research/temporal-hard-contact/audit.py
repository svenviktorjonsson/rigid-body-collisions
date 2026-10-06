"""Independent archived physics/edge/observer checks, retaining real failures."""
import hashlib,json,subprocess
from pathlib import Path
from importlib import import_module
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];load=lambda p:json.loads(p.read_text());sha=lambda raw:hashlib.sha256(raw).hexdigest()
independent=import_module('research.rapid-friction.audit');base=import_module('research.rapid-friction.run');gates=load(ROOT/'research/rapid-friction/plan.json');records=[]
for folder in ['world-results-v2','world-results-v3','world-results-v4']:
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
controls=[]
for folder in ['controls-v2','controls-v3']:
 errors=[]
 for p in sorted((H/folder).glob('work_*.json')):
  d=load(p);r=d['result'];a=np.asarray(r['states']);b=np.asarray(d['observer_disabled_result']['states']);assert a.tobytes()==b.tobytes()
  wall=next(b for b in d['scene']['bodies'] if b.get('type')=='kinematic');first,last=a[0,0],a[-1,0];m=r['mass'][0];I=r['inertia'][0];position=np.asarray(wall['position']);r0=first[:2]-position;r1=last[:2]-position
  cross=lambda u,v:u[0]*v[1]-u[1]*v[0]
  expected=float(np.dot(wall['velocity'],m*(last[3:5]-first[3:5]))+wall['omega']*(I*(last[5]-first[5])+m*(cross(r1,last[3:5])-cross(r0,first[3:5]))))
  error=abs(expected-r['boundary_work_J']);assert error<2e-8;errors.append(error)
 controls.append({'archive':folder,'state_bytes_exact':True,'audited_runs':len(errors),'max_independent_work_error_J':max(errors)})
receipt={'passed':True,'archives':records,'controls':controls,'scope':'Archive consistency verifies original physical gates and actual failed refinement edges, not qualification. Rounded-mass initial build never qualified; approximate-angle polygon authoring failure and procedural driver failures remain retained. Exact rotation/core mass comparator stays isolated.'}
(H/'independent-archive-audit.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print('Temporal archives, unchanged gates, retained failed edges, and independent signed work PASS audit.')
