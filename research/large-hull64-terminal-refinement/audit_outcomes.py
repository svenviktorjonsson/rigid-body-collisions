"""Independent accepted-prefix geometry and actual unchanged-law rejections."""
import ast,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from research.audit_hull_search_completion import authored_progress,ledger,position_projection,trajectory_metrics
from importlib import import_module
H=Path(__file__).resolve().parent;ROOT=H.parents[1];load=lambda p:json.loads(p.read_text());plan=load(H/'plan.json');snapshot=load(ROOT/'research/large-terminal-world/binary-snapshot.json');sha=lambda raw:hashlib.sha256(raw).hexdigest();fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent','exec'),ns);records=[]
for index,setting in enumerate(plan['settings']):
 case={'name':f'trial_{index}','scene_sha256':plan['scene_sha256'],'setting':setting}
 D=H/'results'/case['name'];
 if not (D/'final.json').exists():records.append({'case':case['name'],'pending':True,'reference_qualified':False,'performance_qualified':False});continue
 entry=load(H/'results/scene.json');prov=load(H/'results/provenance.json');live=True
 assert sha(json.dumps(entry['scene'],sort_keys=True,separators=(',',':')).encode())==case['scene_sha256']
 for path,digest in dict(prov['guards'],**prov['runtime']).items():
  p=Path(path);live &= sha(p.read_bytes())==digest
  if p==ROOT/'build/spatial/spatial_runner':raw=Path(snapshot['binaries']['spatial_runner']['path']).read_bytes()
  elif p.parent==ROOT/'spatial_backend' or p==ROOT/'spatial_engine.py':raw=subprocess.check_output(['git','show',plan['source']+':'+str(p.relative_to(ROOT))],cwd=ROOT)
  else:raw=p.read_bytes()
  assert sha(raw)==digest,path
 final=load(D/'final.json');
 if index<len(plan['anchors']):
  anchor=plan['anchors'][index];assert sha((ROOT/anchor['path']).read_bytes())==anchor['sha256'];assert final==load(ROOT/anchor['path'])
 progress=load(D/'progress.json') if (D/'progress.json').exists() else None;
 if final['complete']:
  result=final['result'];metrics=trajectory_metrics(entry['scene'],result);metrics.update(contact_residual_m_s=result['coulomb_residual_max_m_s'],position_residual_m_s=result['translation_split_residual_max_m_s']);limits={'quaternion_norm_error':1e-12,'energy_change_minus_boundary_work_J':1.,'container_surface_excess_m':.002,'contact_residual_m_s':1e-8,'position_residual_m_s':1e-8};assert final['source_binary_unchanged'] and final['setting']==case['setting'] and abs(result['times'][-1]-.12)<1e-12 and np.isfinite(result['states']).all();passed=all(np.isfinite(metrics[k]) and metrics[k]<=v for k,v in limits.items());assert import_module('research.rapid-friction.audit').physical(entry,result)==final['physical']['passed']==passed;records.append({'case':case['name'],'full_history':True,'physical_passed':bool(passed),'metrics':metrics,'source_binary_runtime_verified':True,'current_live_unchanged':bool(live),'reference_qualified':False,'performance_qualified':False});continue
 capture=load(D/'rejection.json');assert not final['complete'] and final['source_binary_unchanged'] and not progress['complete'];assert final['setting']==case['setting'];result,metrics,certificates=authored_progress(entry['scene'],progress,case['setting']['dt']);accounting=ledger(progress);projection=position_projection(progress,1e-8);limits={'quaternion_norm_error':1e-12,'energy_change_minus_boundary_work_J':1.,'container_surface_excess_m':.002};assert all(np.isfinite(metrics[k]) and metrics[k]<=v for k,v in limits.items());assert progress['coulomb_residual_max_m_s']<=1e-8 and progress['translation_split_residual_max_m_s']<=1e-8
 A=np.array(capture['A']);p=np.array(capture['p']);b=np.array(capture['b']);assert np.isfinite(A).all() and np.allclose(A,A.T,rtol=1e-12,atol=1e-12);gate=ns['external'](capture,dict(p=p.tolist(),w=(A@p-b).tolist()));assert not gate['accepted'] and gate['projection_m_s']>1e-8;assert np.isclose(gate['projection_m_s'],capture['residual_m_s'],rtol=1e-7,atol=1e-10);records.append({'case':case['name'],'prefix_simulated_s':progress['times'][-1],'rejected_rows':len(b),'prefix_physical_passed':True,'prefix_metrics':metrics,'basis_certificates':certificates,'pose_ledger':accounting,'position_projection':projection,'original_rejection_gate':gate,'source_binary_runtime_verified':True,'current_live_unchanged':bool(live),'full_history':False,'reference_qualified':False,'performance_qualified':False})
(H/'independent-audit.json').write_text(json.dumps({'passed':True,'records':records,'scope':'Independent real accepted-prefix and actual rejection audit; full history or genuine failure retained, no reference/performance qualification.'},indent=2,allow_nan=False)+'\n');print('Fresh hull independent full-history or accepted-prefix/rejection audits PASS')
