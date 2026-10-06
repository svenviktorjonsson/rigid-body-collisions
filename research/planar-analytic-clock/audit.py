"""Independent full planar archives; separately retain actual native declines."""
import hashlib,json,subprocess
from pathlib import Path
from importlib import import_module
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'world-results';load=lambda p:json.loads(p.read_text());sha=lambda raw:hashlib.sha256(raw).hexdigest();provenance=load(D/'provenance.json');live=True
for path,digest in dict(provenance['guards'],**provenance['runtime']).items():
 p=Path(path);live &= sha(p.read_bytes())==digest
 raw=subprocess.check_output(['git','show',provenance['source']+':'+str(p.relative_to(ROOT))],cwd=ROOT) if p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'build') else p.read_bytes()
 assert sha(raw)==digest,path
assert load(D/'final.json')['complete'];plan=load(H/'plan.json');scenes=load(D/'scenes.json');summary=load(D/'summary.json');independent=import_module('research.rapid-friction.audit');base=import_module('research.rapid-friction.run');gates=load(ROOT/'research/rapid-friction/plan.json');records=[]
for name,entry in scenes.items():
 refs={};declines=[];edges=[]
 for i,setting in enumerate(plan['reference_levels']):
  T=D/name;p=T/f'reference_{i}.json'
  if not p.exists():
   decline=load(T/f'reference_{i}.decline.json');capture=load(T/f'reference_{i}.rejection.json');assert not decline['complete'] and decline['setting']==setting;assert np.isfinite(capture['A']).all() and np.isfinite(capture['p']).all();assert capture['tolerance_m_s']==1e-10;declines.append({'level':i,'reason':capture['reason'],'reported_residual_m_s':capture['residual_m_s'],'scope':'Retained actual native rejection; capture does not contain body velocities needed to independently reproduce body passivity.'});continue
  r=load(p);result=r['result'];assert r['complete'] and r['setting']==setting and np.isfinite(result['states']).all();assert abs(result['times'][-1]-.12)<1e-12;assert independent.physical(entry,result)==r['physical']['passed'];refs[i]=r
  if i-1 in refs:
   err=base.errors(2,refs[i-1]['result'],result);passed=independent.physical(entry,refs[i-1]['result']) and independent.physical(entry,result) and all(err[k]<=v/4 for k,v in gates['trajectory_budgets']['2'].items());edges.append({'left':i-1,'right':i,'passed':bool(passed),'errors':err})
 assert edges==summary[name]['edges'];assert len(refs)==summary[name]['full_histories'];assert sum(independent.physical(entry,r['result']) for r in refs.values())==summary[name]['physical_passes'];last=len(plan['reference_levels'])-1;qualified=len(edges)>=2 and [(e['left'],e['right']) for e in edges[-2:]]==[(last-2,last-1),(last-1,last)] and all(e['passed'] for e in edges[-2:]);assert qualified==summary[name]['reference_qualified'];records.append({'case':name,'full_histories':len(refs),'physical_passes':summary[name]['physical_passes'],'declines':declines,'edges':edges,'reference_qualified':qualified,'performance_qualified':False})
(H/'independent-archive-audit.json').write_text(json.dumps({'passed':True,'historical_guards_verified':True,'current_live_unchanged':bool(live),'records':records,'scope':'Independent full-history physical/accuracy/archive gates. Retained actual body-gate decline is not independently reproduced without its body velocities. No qualification/performance claim.'},indent=2,allow_nan=False)+'\n');print('Global planar full-history independent archive audit PASS')
