"""Basis-invariant mobility-neutral opposing-traction guide, strict final gate."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from research.coulomb_diagnostics import System

def main():
 capture=Path('research/hull-active-completion/results/rejections/fast_shake8_hulls42/reference_1.json');data=json.loads(capture.read_text());s=System.from_dump(data);tol=data['tolerance_m_s'];p=np.array(json.loads(Path('research/active-direct/39-native-homotopy-stalled-base.jsonl').read_text())['numerical_stalled_candidate']);w=s.A@p-s.b;active=[c for c in s.contacts if p[c[0]]>1e-9 or abs(w[c[0]])<10*tol];rows=np.array([r for k,ts,*_ in active for r in (k,*ts)]);_,sv,V=np.linalg.svd(s.A[:,rows],full_matrices=False);rank=int(np.sum(sv>sv[0]*1e-13));N=np.zeros((len(p),len(rows)-rank));N[rows]=V[rank:].T;ks=np.array([c[0] for c in s.contacts]);rn=np.array([c[3] for c in s.contacts]);receipts=[]
 def fb(q,jac=False):
  F,J=s.equations(q,True);u=q[ks]/rn;ww=s.A[ks]@q-s.b[ks];l=np.hypot(u,ww)
  if jac:
   ca=1-np.divide(u,l,out=np.zeros_like(u),where=l>0);cb=1-np.divide(ww,l,out=np.zeros_like(ww),where=l>0);J[ks]=cb[:,None]*s.A[ks];J[ks,ks]+=ca/rn;return J
  F[ks]=u+ww-l;return F
 F=s.equations(p)
 for k,ts,mu,*_ in sorted(active,key=lambda c:np.linalg.norm(F[list(c[1])]),reverse=True):
  t=list(ts);speed=np.linalg.norm(w[t]);target=p.copy();target[t]=-mu*max(0,p[k])*w[t]/speed;direction=N@(N.T@(target-p))
  for fraction in [1.,.5,2.,.25,-1.,-2.]:
   base=p+fraction*direction;initial=base.copy();initial[t]=-mu*max(0,base[k])*w[t]/speed;start=time.perf_counter();r=least_squares(fb,initial,jac=lambda q:fb(q,True),max_nfev=300,ftol=1e-14,xtol=1e-14,gtol=1e-14);gate=s.gate(r.x,tol);q=r.x.copy();q[ks]=np.maximum(0,q[ks]);clipped=s.gate(q,tol)
   record=dict(contact=k,fraction=fraction,neutral_mobility_response_inf=float(np.max(abs(s.A@(base-p)))),neutral_direction=direction.tolist(),neutral_impulse=base.tolist(),initial_impulse=initial.tolist(),impulse=r.x.tolist(),clipped_impulse=q.tolist(),nfev=r.nfev,elapsed_s=time.perf_counter()-start,gate=gate,clipped_gate=clipped);receipts.append(record);print(k,fraction,gate['accepted'],gate['residual_m_s'],r.nfev,flush=True)
   Path(__file__).with_name('native-base-neutral-restarts.json').write_text(json.dumps(dict(capture=str(capture),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),rank=rank,null_dimension=N.shape[1],basis_rows=rows.tolist(),attempts=receipts),indent=2)+'\n')
   if clipped['accepted']:return
if __name__=='__main__':main()
