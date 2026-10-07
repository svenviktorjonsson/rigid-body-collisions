"""Independent exact-geometry Jacobian reconstruction and bounded pose guide.

This guide changes the position policy prospectively; it never qualifies the
original translation-only rejected subproblem or mutates an engine body.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog, minimize


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--geometry',type=Path,required=True);ap.add_argument('--capture',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--translation-bound-m',type=float,default=1e-4);ap.add_argument('--rotation-bound-rad',type=float,default=1e-3)
 a=ap.parse_args();assert not a.output.exists(),'Retain all earlier trials'
 d=json.loads(a.capture.read_text());g=json.loads(a.geometry.read_text());assert g['schema']=='normal-position-geometry-v1'
 rows=g['rows'];bodies={q['solver_body_id']:q for q in g['bodies']};finite=[i for i,q in bodies.items()if q['inverse_mass']>0];index={v:k for k,v in enumerate(finite)}
 n=len(rows);assert n==len(d['b']);h=float(d['internal_dt_s']);assert h==g['internal_dt_s']
 J=np.zeros((n,6*len(finite)));M=np.zeros((6*len(finite),6*len(finite)));targets=np.array([r['split_target_m_s']for r in rows]);assert np.array_equal(targets,np.array(d['b']))
 lever_error=0.
 for k,i in enumerate(finite):
  q=bodies[i];M[6*k:6*k+3,6*k:6*k+3]=np.eye(3)*q['inverse_mass'];M[6*k+3:6*k+6,6*k+3:6*k+6]=np.array(q['inverse_world_inertia'])
  assert q['linear_factor']==[1.,1.,1.] and q['angular_factor']==[1.,1.,1.],'Scoped to unit motion factors'
 for i,r in enumerate(rows):
  assert r['normal_row_index']==i and r['has_manifold_point']
  for side in ['a','b']:
   j=r['solver_body_id_'+side];q=bodies[j];normal=np.array(r['linear_jacobian_'+side]);angular=np.array(r['angular_jacobian_'+side]);point=np.array(r['shared_world_point']);center=np.array(q['solver_transform']['position'])
   if q['has_original_body']:lever_error=max(lever_error,float(np.linalg.norm(np.cross(point-center,normal)-angular)))
   if j in index:k=index[j];J[i,6*k:6*k+3]=normal;J[i,6*k+3:6*k+6]=angular
 T=J.copy();T[:,3::6]=0;T[:,4::6]=0;T[:,5::6]=0;Atrans=T@M@T.T;Af=J@M@J.T;err=float(np.max(abs(Atrans-np.array(d['A']))));assert err<1e-12,'Geometry does not reconstruct actual frozen translation matrix'
 cert=json.loads(Path('research/translation-position-certificate/witness.json').read_text());weights=np.array([int(v['numerator'])/int(v['denominator'])for v in cert['weights']]);support=cert['support'];wJ=weights@J[support];wT=weights@T[support]
 out=dict(capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest(),geometry_sha256=hashlib.sha256(a.geometry.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Prospective bounded full-pose linear guide only; independent nonlinear contact re-query required before integration',translation_reconstruction_max=err,lever_reconstruction_error_m=lever_error,support=support,support_rows=[rows[i]for i in support],weighted_translation_J_norm=float(np.linalg.norm(wT)),weighted_full_J_norm=float(np.linalg.norm(wJ)),weighted_target_m_s=float(weights@targets[support]),translation_rank=int(np.linalg.matrix_rank(Atrans)),full_rank=int(np.linalg.matrix_rank(Af)),finite_body_count=len(finite),finite_body_ids=[bodies[i]['body_id']for i in finite],attempts=[])
 # Bound every component: actual Euclidean displacement must be gated separately.
 # Variables are scaled to metres for conditioning; angular coordinates multiply
 # the declared reference length, never change the physical Jacobian or target.
 length=.1;scale=np.tile([1,1,1,1/length,1/length,1/length],len(finite));K=J*scale;bound=np.tile([a.translation_bound_m]*3+[length*a.rotation_bound_rad]*3,len(finite));target=h*targets
 r=linprog(np.zeros(K.shape[1]),A_ub=-K,b_ub=-target,bounds=list(zip(-bound,bound)),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
 z=r.x if r.x is not None else np.zeros(K.shape[1]);Q=np.linalg.pinv(M,rcond=1e-14)*scale[:,None]*scale[None,:];Q/=max(np.max(np.diag(Q)),1e-30)
 for label,solver in [('bounded-feasibility-LP',r),('bounded-minimum-pose-QP',minimize(lambda z:.5*z@Q@z,z,jac=lambda z:Q@z,bounds=list(zip(-bound,bound)),constraints=[dict(type='ineq',fun=lambda z:K@z-target,jac=lambda z:K)],method='SLSQP',options={'ftol':1e-20,'maxiter':2000}))]:
  z=solver.x if solver.x is not None else np.zeros(K.shape[1]);q=scale*z;violation=float(max(0,np.max(target-K@z))/h);vec=q.reshape(-1,6)
  rec=dict(method=label,optimizer_success=bool(solver.success),message=str(solver.message),linear_inward_violation_m_s=violation,original_tolerance_m_s=d['tolerance_m_s'],linear_qualified=bool(violation<=d['tolerance_m_s']),max_translation_norm_m=float(np.max(np.linalg.norm(vec[:,:3],axis=1))),max_rotation_norm_rad=float(np.max(np.linalg.norm(vec[:,3:],axis=1))),pose_increment=vec.tolist(),nonlinear_qualified=False,nonlinear_reason='Actual convex gap re-query not performed; linear qualification alone is insufficient')
  out['attempts'].append(rec)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in ['support_rows','attempts']},indent=2));print([(q['method'],q['linear_qualified'],q['linear_inward_violation_m_s'])for q in out['attempts']])
if __name__=='__main__':main()
