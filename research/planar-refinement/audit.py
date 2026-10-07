"""Independent archived geometry, energy, refinement and default byte checks."""
import json
from pathlib import Path
import numpy as np
from importlib import import_module
_audit=import_module('research.rapid-friction.audit');physical=_audit.physical;errors=_audit.errors
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'results'
load=lambda p:json.loads(p.read_text())
assert load(D/'final.json')['complete'] and load(D/'final.json')['source_unchanged']
scene=load(D/'scenes.json')['2d_mixed_polygons'];budget=load(ROOT/'research/rapid-friction/plan.json')['trajectory_budgets']['2'];summary=load(D/'summary.json')['2d_mixed_polygons'];records=[]
for phase in summary['phases']:
 name=phase['phase']['id'];refs=[]
 for i in range(5):
  r=load(D/'2d_mixed_polygons'/f'{name}_reference_{i}.json');assert physical(scene,r['result'])==r['physical']['passed'];refs.append(r)
  if name=='seams_off' and i==0:
   old=load(ROOT/'research/rapid-friction/results-planar-discovery/2d_mixed_polygons'/'seams_off_reference_4.json')
   assert np.array(old['result']['states'],dtype=np.float64).tobytes()==np.array(r['result']['states'],dtype=np.float64).tobytes()
  records.append({'phase':name,'level':i,'physical_passed':r['physical']['passed']})
 for i,e in enumerate(phase['edges']):
  a,b=refs[i:i+2];delta=errors(2,a['result'],b['result']);passed=a['physical']['passed'] and b['physical']['passed'] and all(delta[k]<=v/4 for k,v in budget.items());assert passed==e['passed']
  for k in delta:assert np.isclose(delta[k],e['errors'][k],rtol=1e-10,atol=1e-12)
 assert phase['qualified']==all(e['passed'] for e in phase['edges'][-2:])
assert not summary['reference_qualified']
(H/'independent-audit.json').write_text(json.dumps({'passed':True,'historical_finest_replay_states_exact':True,'records':records,'qualified':False,'adopted':False,'scope':'Audit passes by verifying retained failures, not by qualifying the prototype.'},indent=2)+'\n')
print('Finer planar archive audit PASS; all original refinement edges decline.')
