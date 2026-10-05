"""Prospective signed-gap position discretization; original capture unchanged."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,least_squares
P=Path(__file__).parent;D=P/'results';out=D/'gap-aware-position-guides.json';assert not out.exists()
capture=D/'rejected-normal-system.json';geometry=D/'rejected-normal-system.json.geometry.json';d=json.loads(capture.read_text());g=json.loads(geometry.read_text());A=np.array(d['A']);oldb=np.array(d['b']);h=d['internal_dt_s'];tol=d['tolerance_m_s'];hi=np.array(d['hi']);slop=1e-9
rows=g['rows'];distance=np.array([r['signed_distance_m']for r in rows]);b=np.where(distance>slop,-(distance-slop)/h,oldb)
bodies={q['solver_body_id']:q for q in g['bodies']};finite=[i for i,q in bodies.items()if q['inverse_mass']>0];indices={v:k for k,v in enumerate(finite)};J=np.zeros((len(b),3*len(finite)));M=np.zeros((3*len(finite),3*len(finite)))
for k,i in enumerate(finite):M[3*k:3*k+3,3*k:3*k+3]=np.eye(3)*bodies[i]['inverse_mass']
for i,r in enumerate(rows):
 for side in ['a','b']:
  j=r['solver_body_id_'+side]
  if j in indices:k=indices[j];J[i,3*k:3*k+3]=r['linear_jacobian_'+side]
assert np.max(abs(J@M@J.T-A))<1e-12
rho=1/np.diag(A)
def gate(p,b):
 w=A@p-b;res=float(np.max(abs(p-np.maximum(0,p-rho*w))/rho));energy=float(.5*p@A@p-b@p);scale=float(1+np.sum(abs(p*b)));valid=bool(np.isfinite(p).all()and np.isfinite(w).all()and np.isfinite(energy)and np.min(p)>=0 and np.all(p<=hi)and res<=tol and energy<=tol*scale)
 return dict(accepted=valid,residual_m_s=res,min_normal_velocity_m_s=float(np.min(w)),passive_change_bound_J=energy,pressure_max_Ns=float(np.max(p)))
def fb(p,jac=False):
 u=p/rho;w=A@p-b;l=np.hypot(u,w)
 if jac:
  ca=1-np.divide(u,l,out=np.zeros_like(u),where=l>0);cb=1-np.divide(w,l,out=np.zeros_like(w),where=l>0);T=cb[:,None]*A;T[np.diag_indices_from(T)]+=ca/rho;return T
 return u+w-l
report=dict(schema='prospective-gap-aware-position-probe-v1',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),declared_change='Positive-gap normal position rows may consume distance minus declared1e-9m slop. Penetrating bSplit, A, hi, tolerance and materials unchanged.',original_target=oldb.tolist(),prospective_target=b.tolist(),gap_slop_m=slop,finite_body_ids=[bodies[i]['body_id']for i in finite],attempts=[])
for method in ['pressure-QP-cold','normal-FB-cold']:
 start=time.perf_counter();p=np.zeros(len(b))
 if method=='pressure-QP-cold':r=minimize(lambda p:.5*p@A@p-b@p,p,jac=lambda p:A@p-b,bounds=[(0,None)]*len(b),method='SLSQP',options={'ftol':1e-24,'maxiter':2000})
 else:r=least_squares(fb,p,jac=lambda p:fb(p,True),max_nfev=2000,ftol=1e-14,xtol=1e-14,gtol=1e-14)
 raw=r.x;p=np.maximum(raw,0);delta=h*M@J.T@p;vec=np.zeros((len(finite),6));vec[:,:3]=delta.reshape(-1,3)
 trial=dict(method=method,optimizer_success=bool(r.success),message=str(r.message),nfev=int(r.nfev),iterations=int(getattr(r,'nit',0)),elapsed_s=time.perf_counter()-start,raw_impulse=raw.tolist(),impulse=p.tolist(),original_gate=gate(p,oldb),prospective_gate=gate(p,b),pose_increment=vec.tolist(),max_translation_norm_m=float(np.max(np.linalg.norm(vec[:,:3],axis=1))),max_rotation_norm_rad=0.,linear_qualified=gate(p,b)['accepted'],nonlinear_qualified=False,trajectory_qualified=False)
 report['attempts'].append(trial);out.write_text(json.dumps(report,indent=2)+'\n');print(method,trial['prospective_gate'],trial['max_translation_norm_m'],flush=True)
