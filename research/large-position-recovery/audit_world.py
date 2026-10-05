"""Independent retained-world geometry, original gates and provenance audit."""
import hashlib,json,subprocess
from importlib import import_module
from pathlib import Path
import numpy as np
from research.audit_hull_search_completion import trajectory_metrics
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'world-results';load=lambda p:json.loads(p.read_text())
plan=load(H/'world-plan.json');record=load(D/'final.json');entry=load(D/'scene.json');provenance=load(D/'provenance.json');r=record['result'];assert record['complete'] and record['source_binary_unchanged']
assert hashlib.sha256(json.dumps(entry['scene'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==plan['scene_sha256'];assert record['setting']==plan['setting'] and len(entry['scene']['bodies'])-1==125
assert np.isfinite(np.asarray(r['states'])).all() and abs(r['times'][-1]-.12)<1e-12 and r['collision_updates']==192000
snapshot=load(H/'world-binary-snapshot.json');live_unchanged=True
for path,sha in dict(provenance['guards'],**provenance['runtime']).items():
 p=Path(path);live_unchanged &= hashlib.sha256(p.read_bytes()).hexdigest()==sha
 if p==ROOT/'build/spatial/spatial_runner':raw=Path(snapshot['path']).read_bytes()
 elif p.parent==ROOT/'spatial_backend' or p==ROOT/'spatial_engine.py':raw=subprocess.check_output(['git','show',plan['source']+':'+str(p.relative_to(ROOT))],cwd=ROOT)
 else:raw=p.read_bytes()
 assert hashlib.sha256(raw).hexdigest()==sha,path
assert provenance['integrated_source']==plan['source']
metrics=trajectory_metrics(entry['scene'],r);metrics.update(contact_residual_m_s=r['coulomb_residual_max_m_s'],position_residual_m_s=r['translation_split_residual_max_m_s'])
limits={'quaternion_norm_error':1e-12,'energy_change_minus_boundary_work_J':1.,'container_surface_excess_m':.002,'contact_residual_m_s':1e-8,'position_residual_m_s':1e-8};assert all(np.isfinite(metrics[k]) and metrics[k]<=v for k,v in limits.items())
assert import_module('research.rapid-friction.audit').physical(entry,r)==record['physical']['passed']==True
assert r['numerical_model']['position_normal_search_policy']['max_normal_rows']==512
receipt={'passed':True,'historical_guards_verified':True,'current_live_unchanged':bool(live_unchanged),'complete_full_horizon':True,'body_count':125,'updates':192000,'simulated_s':.12,'scene_sha256':plan['scene_sha256'],'source':plan['source'],'original_physical_gates_passed':True,'metrics':metrics,'limits':limits,'reference_qualified':False,'performance_qualified':False,'scope':'Single completed original125-hull finest history. Independent archived geometry/energy and original gate checks; source/binary/runtime guards match. Concurrent research and brief sampling make elapsed descriptive only; both adjacent refinement edges still required.'}
(H/'world-independent-recheck.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='metrics'},indent=2))
