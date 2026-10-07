"""Recompute gap-law targets, all-row gates, pose and provenance independently."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
import numpy as np
P=Path(__file__).parent;ROOT=P.parents[1];D=P/'results';dest=D/'independent-audit.json';sha=lambda raw:hashlib.sha256(raw).hexdigest();prov=json.loads((D/'provenance.json').read_text());plan=json.loads((P/'plan.json').read_text())
with zipfile.ZipFile(D/'source.zip')as z:
 for p,expected in prov['source_hashes'].items():assert sha(z.read(p))==expected and z.read(p)==subprocess.check_output(['git','show',prov['execution_source_commit']+':'+p],cwd=ROOT),p
 cap=z.read(plan['capture']);geom=z.read(plan['geometry']);assert sha(cap)==plan['capture_sha256']==prov['capture_sha256']and sha(geom)==plan['geometry_sha256']==prov['geometry_sha256']
c=json.loads(cap);g=json.loads(geom);A=np.array(c['A']);b=np.array(c['b']);h=c['internal_dt_s'];tol=c['tolerance_m_s'];distance=np.array([r['signed_distance_m']for r in g['rows']]);target=b.copy()
for i,value in enumerate(distance):
 if value>plan['contact_slop_m']:target[i]=-(value-plan['contact_slop_m'])/h
assert np.array_equal(np.array(json.loads((D/'summary.json').read_text())['prospective_target']),target)
summary=json.loads((D/'summary.json').read_text());q=json.loads((D/'geometry-requery.json').read_text());body={v['body_id']:v for v in g['bodies']if v['has_original_body']};solver_body={v['solver_body_id']:v for v in g['bodies']};records=[]
def pairs(q):
 values={}
 for c in q['contacts']:
  k=tuple(sorted([c['body_a'],c['body_b']]));values[k]=min(values.get(k,float('inf')),c['signed_distance_m'])
 return values
before=pairs(q['before'])
for t,r in zip(summary['attempts'],q['trials']):
 p=np.array(t['impulse']);w=A@p-target;res=float(np.max(abs(p-np.maximum(0,p-w/np.diag(A)))*np.diag(A)));energy=float(.5*p@A@p-target@p);scale=float(1+np.sum(abs(p*target)));finite=bool(np.isfinite(p).all()and np.isfinite(w).all()and np.isfinite(energy)and np.isfinite(scale));bounds=bool(np.min(p)>=0 and np.all(p<=c['hi']));mechanical=finite and bounds and res<=tol and np.min(w)>=-tol and energy<=tol*scale
 expected={ident:np.zeros(3)for ident in summary['finite_body_ids']}
 for i,row in enumerate(g['rows']):
  for side in ['a','b']:
   v=solver_body[row['solver_body_id_'+side]]
   if v['inverse_mass']>0:expected[v['body_id']]+=h*v['inverse_mass']*p[i]*np.array(row['linear_jacobian_'+side])
 maxerr=0.;M=0.;com=np.zeros(3);L=np.zeros(3);U=0.;gravity=np.array(plan['gravity_m_s2'])
 for ident,pose in zip(summary['finite_body_ids'],t['pose_increment']):
  delta=np.array(pose[:3]);assert pose[3:]==[0.,0.,0.];maxerr=max(maxerr,float(np.max(abs(delta-expected[ident]))));v=body[ident];m=v['mass'];M+=m;com+=m*delta;L+=np.cross(delta,m*np.array(v['linear_velocity']));U-=m*np.dot(gravity,delta)
 assert maxerr<1e-18;assert np.max(abs(com/M-t['ledger']['COM_change_m']))<1e-20;assert np.max(abs(L-t['ledger']['total_angular_momentum_change_kg_m2_s']))<1e-18;assert abs(U-t['ledger']['gravity_potential_change_J'])<1e-20
 after=pairs(r['after']);excess=max([0.]+[min(before.get(k,0.),0.)-v for k,v in after.items()]);shape=excess<=plan['contact_slop_m'];passed=bool(mechanical and shape and t['max_translation_norm_m']<=plan['pose_bound_m']);assert passed==t['prospective_trial_qualified'];records.append(dict(method=t['method'],passed=passed,mechanical_passed=mechanical,residual_m_s=res,bounds_passed=bounds,passive_bound_J=energy,pose_reconstruction_error_m=maxerr,pair_worsening_m=excess,geometry_passed=shape,gravity_potential_change_J=U))
result=dict(passed=True,execution_source_commit=prov['execution_source_commit'],capture_sha256=prov['capture_sha256'],geometry_sha256=prov['geometry_sha256'],qualified_prospective_trials=sum(r['passed']for r in records),retained_failed_trials=sum(not r['passed']for r in records),trials=records,original_policy_qualified=False,trajectory_qualified=False)
if dest.exists():
 old=json.loads(dest.read_text());assert old['execution_source_commit']==result['execution_source_commit'] and old['capture_sha256']==result['capture_sha256']and old['geometry_sha256']==result['geometry_sha256'];assert [(t['method'],t['passed'])for t in old['trials']]==[(t['method'],t['passed'])for t in records]
else:dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
