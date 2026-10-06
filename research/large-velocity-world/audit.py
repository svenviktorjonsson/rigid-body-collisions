"""Independent geometry/inertia of accepted prefixes and real rejection laws."""
import ast,hashlib,json,shutil
from pathlib import Path
import numpy as np
from research.audit_hull_search_completion import authored_progress,ledger,position_projection
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'results';load=lambda p:json.loads(p.read_text());plan=load(H/'plan.json');entry=load(D/'scene.json');provenance=load(D/'provenance.json');final=load(D/'final.json');assert final['complete']
assert hashlib.sha256(json.dumps(entry['scene'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==plan['scene_sha256']
for path,sha in dict(provenance['guards'],**provenance['runtime']).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
# Preserve executable before later numerical changes; initial live attestation remains immutable.
exe=ROOT/'build/spatial/spatial_runner';target=Path('/home/viktor/.cache/physics-null-seed64-reference-binaries-20261006');target.mkdir(exist_ok=True);shutil.copy2(exe,target/'spatial_runner');(H/'binary-snapshot.json').write_text(json.dumps({'path':str(target/'spatial_runner'),'sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'source':plan['source']},indent=2)+'\n')
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent','exec'),ns);records=[]
for i,setting in enumerate(plan['settings']):
 T=D/f'trial_{i}';record=load(T/'final.json');progress=load(T/'progress.json');capture=load(T/'rejection.json');assert not record['complete'] and record['source_binary_unchanged'] and not progress['complete'];result,metrics,certificates=authored_progress(entry['scene'],progress,setting['dt']);accounting=ledger(progress);projection=position_projection(progress,1e-8)
 limits={'quaternion_norm_error':1e-12,'energy_change_minus_boundary_work_J':1.,'container_surface_excess_m':.002};assert all(np.isfinite(metrics[k]) and metrics[k]<=v for k,v in limits.items());assert progress['coulomb_residual_max_m_s']<=1e-8 and progress['translation_split_residual_max_m_s']<=1e-8
 A=np.array(capture['A']);p=np.array(capture['p']);b=np.array(capture['b']);assert np.isfinite(A).all() and np.allclose(A,A.T,rtol=1e-12,atol=1e-12);check=ns['external'](capture,{'p':p.tolist(),'w':(A@p-b).tolist()});assert not check['accepted'] and check['projection_m_s']>1e-8;assert np.isclose(check['projection_m_s'],capture['residual_m_s'],rtol=1e-7,atol=1e-10)
 records.append({'trial':i,'setting':setting,'rejected_rows':len(b),'completed_output_frames':progress['completed_output_frames'],'prefix_simulated_s':progress['times'][-1],'prefix_physical_gates_passed':True,'prefix_metrics':metrics,'basis_certificates':certificates,'pose_ledger':accounting,'position_projection':projection,'rejection_original_law':check,'reference_qualified':False,'performance_qualified':False})
(H/'independent-audit.json').write_text(json.dumps({'passed':True,'source_runtime_unchanged':True,'records':records,'scope':'Verified accepted prefixes and genuine later original-law declines; not full trajectory qualification.'},indent=2)+'\n');print('Both accepted-prefix geometry/inertia/ledger and later rejection audits PASS; no full histories.')
