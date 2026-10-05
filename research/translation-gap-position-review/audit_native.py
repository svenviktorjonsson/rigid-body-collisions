"""Independent retained native signed-gap source, all-row and geometry audit."""
import json,hashlib,subprocess,zipfile
from pathlib import Path
import numpy as np
P=Path(__file__).parent;ROOT=P.parents[1];D=P/'native-results';sha=lambda raw:hashlib.sha256(raw).hexdigest();prov=json.loads((D/'provenance.json').read_text());plan=json.loads((P/'plan.json').read_text());native=json.loads((D/'native-receipt.json').read_text());published=json.loads((D/'independent-native-audit.json').read_text());q=json.loads((D/'geometry-requery.json').read_text())
with zipfile.ZipFile(D/'source.zip')as z:
 counts={}
 for info in z.infolist():
  raw=z.read(info);counts[info.filename]=counts.get(info.filename,0)+1
  if info.filename in prov['source_sha256']:
   assert sha(raw)==prov['source_sha256'][info.filename];assert raw==subprocess.check_output(['git','show',prov['execution_source_commit']+':'+info.filename],cwd=ROOT)
 for p,origin in prov['frozen_header_origin'].items():assert z.read(p)==subprocess.check_output(['git','show',origin['source_commit']+':'+origin['original_path']],cwd=ROOT)
 cap=z.read(plan['capture']);geom=z.read(plan['geometry']);assert sha(cap)==prov['capture_sha256']==plan['capture_sha256']and sha(geom)==prov['geometry_sha256']==plan['geometry_sha256']
d=json.loads(cap);g=json.loads(geom);A=np.array(d['A']);b=np.array(d['b']);target=np.array([-(r['signed_distance_m']-plan['contact_slop_m'])/d['internal_dt_s']if r['signed_distance_m']>plan['contact_slop_m']else b[i]for i,r in enumerate(g['rows'])]);assert np.array_equal(target,np.array(native['prospective_target']));t=native['attempts'][0];p=np.array(t['impulse']);w=A@p-target;error=float(np.max(abs(p-np.maximum(0,p-w/np.diag(A)))*np.diag(A)));energy=float(.5*p@A@p-target@p);scale=float(1+np.sum(abs(p*target)));tol=d['tolerance_m_s'];mechanical=bool(np.isfinite(p).all()and np.isfinite(w).all()and np.isfinite(energy)and np.isfinite(scale)and np.min(p)>=0 and np.all(p<=d['hi'])and error<=tol and np.min(w)>=-tol and energy<=tol*scale)
body={b['solver_body_id']:b for b in g['bodies']};delta={ident:np.zeros(3)for ident in native['finite_body_ids']}
for i,r in enumerate(g['rows']):
 for side in ['a','b']:
  v=body[r['solver_body_id_'+side]]
  if v['inverse_mass']>0:delta[v['body_id']]+=d['internal_dt_s']*v['inverse_mass']*p[i]*np.array(r['linear_jacobian_'+side])
maxerr=0.
for ident,pose in zip(native['finite_body_ids'],t['pose_increment']):assert pose[3:]==[0.,0.,0.];maxerr=max(maxerr,float(np.max(abs(np.array(pose[:3])-delta[ident]))))
assert maxerr<1e-18

def pairs(q):
 out={}
 for c in q['contacts']:
  pair=tuple(sorted([c['body_a'],c['body_b']]));out[pair]=min(out.get(pair,float('inf')),c['signed_distance_m'])
 return out
before=pairs(q['before']);after=pairs(q['trials'][0]['after']);worsening=max([0.]+[min(before.get(k,0.),0.)-v for k,v in after.items()]);passed=bool(mechanical and native['native_solved']and not native['original_native_solved']and worsening<=plan['contact_slop_m']and t['max_translation_norm_m']<=plan['pose_bound_m']);assert passed==published['passed']
result=dict(passed=passed,execution_source_commit=prov['execution_source_commit'],existing_solver_header_source='c465f8556ae905f69908eb3fb1427db50737d244',original_declined=True,all74_residual_m_s=error,passive_correction_bound_J=energy,pose_reconstruction_error_m=maxerr,pair_worsening_m=worsening,duplicate_archive_entries={k:v for k,v in counts.items()if v>1},duplicates_byte_verified=True,trajectory_qualified=False)
f=D/'repeatable-independent-audit.json'
if f.exists():assert json.loads(f.read_text())['passed']==result['passed']
else:f.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
