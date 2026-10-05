"""Bounded numerical-J experiments on a rejected immutable shared-contact dump."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import lstsq
from research.coulomb_diagnostics import System,recover,analyze
from research.hull_recovery_followup import opposing_slip_restart,serial
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'research/shared-hull-diagnostics'
INPUT=ROOT/'research/shared-hull-followup/results/rejections/fast_shake8_hulls42/reference_1.json'
SOURCE='38f407d12208654075c07315e96b9fc213612b91'

def physical_gate(S,p,tolerance):
 p=np.asarray(p);w=S.A@p-S.b;errors=[]
 for k,ts,mu,rn,rt in S.contacts:
  t=list(ts);z=p[t]-rt*w[t];length=np.linalg.norm(z);cap=mu*max(p[k],0.)
  proj=z if length<=cap else cap*z/length if length else np.zeros(2)
  errors.extend([abs(min(p[k]/rn,w[k])),np.linalg.norm((p[t]-proj)/rt)])
 residual=float(max(errors));work=float(.5*p@S.A@p-S.b@p);scale=float(1+np.sum(abs(p*S.b)))
 return dict(accepted=bool(np.isfinite(p).all() and np.isfinite(work) and np.isfinite(scale) and residual<=tolerance and work<=tolerance*scale),residual_m_s=residual,passive_change_bound_J=work)

def main():
 data=json.loads(INPUT.read_text());S=System.from_dump(data);p=np.asarray(data['p']);tol=data['tolerance_m_s'];F,J=S.equations(p,True);U,s,V=np.linalg.svd(J)
 experiments=[]
 for cutoff in [1e-13,1e-12,1e-10,1e-9,1e-8]:
  step,_,rank,_=lstsq(J,-F,cond=cutoff,lapack_driver='gelsd');q=p+step
  experiments.append(dict(kind='truncated-J-increment',relative_cutoff=cutoff,rank=int(rank),step_norm_N_s=float(np.linalg.norm(step)),independent_gate=physical_gate(S,q,tol),impulse=q.tolist()))
 for relative in [1e-12,1e-10,1e-8,1e-7,1e-6,1e-5,1e-4]:
  lam=relative*s[0];step=V.T@((s/(s*s+lam*lam))*(U.T@(-F)));q=p+step
  experiments.append(dict(kind='damped-J-increment',relative_lambda=relative,absolute_lambda=float(lam),step_norm_N_s=float(np.linalg.norm(step)),independent_gate=physical_gate(S,q,tol),impulse=q.tolist()))
 # All final gates use the original A, b and mu. A truncation is neither a
 # physical compliance term nor a modified contact target.
 assert next(e for e in experiments if e.get('relative_cutoff')==1e-10)['independent_gate']['accepted']
 assert not next(e for e in experiments if e.get('relative_cutoff')==1e-12)['independent_gate']['accepted']
 r=recover(S,p,tol);opposing=opposing_slip_restart(S,p,tol)
 result=dict(execution_input_source=SOURCE,input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
             input_path=str(INPUT.relative_to(ROOT)),J_singular_values=s.tolist(),singular_ratio=float(s[-1]/s[0]),
             experiments=experiments,published_recovery=serial(r),opposing_restart=serial(opposing),analysis=analyze(S,p,tol))
 DIRECTORY.mkdir(exist_ok=True);(DIRECTORY/'bounded-experiments.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 (DIRECTORY/'seed42-48row-linear-input.json').write_text(json.dumps(dict(matrix=J.ravel().tolist(),rhs=(-F).tolist()),indent=2)+'\n')
 print('48-row unchanged-law gates: rank47 one-step accepted; rank48 full-step rejected; all12 bounded variants retained')
if __name__=='__main__':main()
