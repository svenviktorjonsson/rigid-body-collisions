"""Independent full histories, true declines and original adjacent accuracy gates."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from research.audit_hull_search_completion import trajectory_metrics,authored_progress,ledger,position_projection
ROOT=Path(__file__).resolve().parents[1];H=ROOT/sys.argv[1];plan=json.loads((H/'plan.json').read_text());load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();original=import_module('research.rapid-friction.audit');metrics_module=import_module('research.rapid-friction.run');gates=load(ROOT/'research/rapid-friction/plan.json');snapshot=load(ROOT/'research/large-terminal-world/binary-snapshot.json');records=[]
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'original gate','exec'),ns)
for name in plan['cases']:
 D=H/'results'/name;prov=load(D/'provenance.json');assert load(D/'final.json')['complete'];entry=load(D/'scene.json');assert hashlib.sha256(json.dumps(entry['scene'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==plan['cases'][name];live=True
 for path,digest in dict(prov['guards'],**prov['runtime']).items():
  p=Path(path);live &= sha(p)==digest
  if p==ROOT/'build/spatial/spatial_runner':raw=Path(snapshot['binaries']['spatial_runner']['path']).read_bytes()
  elif p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'build'):raw=subprocess.check_output(['git','show',prov['execution_source']+':'+str(p.relative_to(ROOT))],cwd=ROOT)
  else:raw=p.read_bytes()
  assert hashlib.sha256(raw).hexdigest()==digest,path
 refs={};history=[];edges=[]
 for i,setting in enumerate(plan['settings']):
  T=D/f'trial_{i}';r=load(T/'final.json');assert r['source_binary_unchanged'] and r['setting']==setting
  if r['complete']:
   result=r['result'];assert np.isfinite(result['states']).all() and abs(result['times'][-1]-.12)<1e-12;metrics=trajectory_metrics(entry['scene'],result);metrics.update(contact_residual_m_s=result['coulomb_residual_max_m_s'],position_residual_m_s=result['translation_split_residual_max_m_s']);limits={'quaternion_norm_error':1e-12,'energy_change_minus_boundary_work_J':1.,'container_surface_excess_m':.002,'contact_residual_m_s':1e-8,'position_residual_m_s':1e-8};passed=all(np.isfinite(metrics[k]) and metrics[k]<=v for k,v in limits.items());assert original.physical(entry,result)==r['physical']['passed']==passed;refs[i]=r;history.append({'trial':i,'complete':True,'physical_passed':bool(passed),'metrics':metrics})
   if i-1 in refs:
    errors=metrics_module.errors(3,refs[i-1]['result'],result);passed=refs[i-1]['physical']['passed'] and r['physical']['passed'] and all(errors[k]<=v/4 for k,v in gates['trajectory_budgets']['3'].items());edges.append({'left':i-1,'right':i,'passed':bool(passed),'errors':errors})
  else:
   capture=load(T/'rejection.json');p=np.array(capture['p']);A=np.array(capture['A']);b=np.array(capture['b']);assert np.isfinite(A).all() and np.allclose(A,A.T,rtol=1e-12,atol=1e-12);gate=ns['external'](capture,{'p':p.tolist(),'w':(A@p-b).tolist()});assert not gate['accepted'];assert np.isclose(gate['projection_m_s'],capture['residual_m_s'],rtol=1e-7,atol=1e-10);progress=load(T/'progress.json');prefix,metrics,certificates=authored_progress(entry['scene'],progress,setting['dt']);assert progress['coulomb_residual_max_m_s']<=1e-8 and progress['translation_split_residual_max_m_s']<=1e-8;assert all(np.isfinite(metrics[k]) and metrics[k]<=v for k,v in {'quaternion_norm_error':1e-12,'energy_change_minus_boundary_work_J':1.,'container_surface_excess_m':.002}.items());history.append({'trial':i,'complete':False,'prefix_physical_passed':True,'prefix_s':progress['times'][-1],'rejection_rows':len(b),'original_rejection_gate':gate,'pose_ledger':ledger(progress),'position_projection':position_projection(progress,1e-8),'basis_certificates':certificates})
 if (D/'edges.json').exists():assert load(D/'edges.json')==edges
 last=len(plan['settings'])-1;qualified=len(edges)>=2 and [(e['left'],e['right']) for e in edges[-2:]]==[(last-2,last-1),(last-1,last)] and all(e['passed'] for e in edges[-2:]);records.append({'case':name,'records':history,'edges':edges,'reference_qualified':qualified,'performance_qualified':False,'historical_guards_verified':True,'current_live_unchanged':bool(live)})
(H/'independent-audit.json').write_text(json.dumps({'passed':True,'records':records,'scope':'Independent retained outcomes/full physics/actual declines/BOTH original quarter-budget edges. No baseline/performance acceptance.'},indent=2,allow_nan=False)+'\n');print('Spatial independent archive audit PASS',[(r['case'],r['reference_qualified']) for r in records])
