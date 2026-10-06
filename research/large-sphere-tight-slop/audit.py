"""Independent full sphere histories, frozen provenance and both required edges."""
import hashlib,json,subprocess
from pathlib import Path
from importlib import import_module
import numpy as np
from research.audit_hull_search_completion import trajectory_metrics
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'results';load=lambda p:json.loads(p.read_text());plan=load(H/'plan.json');provenance=load(D/'provenance.json');entry=load(D/'scene.json');snapshot=load(H/'binary-snapshot.json');live=True
for path,sha in dict(provenance['guards'],**provenance['runtime']).items():
 p=Path(path);live &= hashlib.sha256(p.read_bytes()).hexdigest()==sha
 if p==ROOT/'build/spatial/spatial_runner':raw=Path(snapshot['path']).read_bytes()
 elif p.parent==ROOT/'spatial_backend' or p==ROOT/'spatial_engine.py':raw=subprocess.check_output(['git','show',plan['source']+':'+str(p.relative_to(ROOT))],cwd=ROOT)
 else:raw=p.read_bytes()
 assert hashlib.sha256(raw).hexdigest()==sha,path
assert load(D/'final.json')['complete'];limits={'quaternion_norm_error':1e-12,'energy_change_minus_boundary_work_J':1.,'container_surface_excess_m':.002,'contact_residual_m_s':1e-8,'position_residual_m_s':1e-8};records=[];refs=[]
for i,setting in enumerate(plan['settings']):
 r=load(D/f'trial_{i}/final.json');result=r['result'];assert r['complete'] and r['source_binary_unchanged'] and r['setting']==setting
 assert np.isfinite(result['states']).all() and abs(result['times'][-1]-.12)<1e-12
 metrics=trajectory_metrics(entry['scene'],result);metrics.update(contact_residual_m_s=result['coulomb_residual_max_m_s'],position_residual_m_s=result['translation_split_residual_max_m_s']);assert all(np.isfinite(metrics[k]) and metrics[k]<=v for k,v in limits.items())
 assert import_module('research.rapid-friction.audit').physical(entry,result)==r['physical']['passed']==True
 records.append({'trial':i,'setting':setting,'metrics':metrics,'physical_passed':True});refs.append(result)
base=import_module('research.rapid-friction.large_irregular');budget=load(ROOT/'research/rapid-friction/plan.json')['trajectory_budgets']['3'];edges=[]
for i,(a,b) in enumerate(zip(refs,refs[1:])):
 error=base.errors(3,a,b);edges.append({'left':i,'right':i+1,'errors':error,'quarter_budget_limits':{k:v/4 for k,v in budget.items()},'passed':all(error[k]<=v/4 for k,v in budget.items())})
qualified=all(e['passed'] for e in edges[-2:]);receipt={'passed':True,'historical_guards_verified':True,'current_live_unchanged':bool(live),'records':records,'edges':edges,'reference_qualified':qualified,'performance_qualified':False,'scope':'Three full unchanged125-sphere histories pass independent original physical gates. BOTH adjacent quarter-budget edges required; actual failed spin edge remains retained. Timing descriptive only.'}
(H/'independent-audit.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print('Sphere physical/archive audit PASS; both edges',[(e['passed'],e['errors']) for e in edges])
