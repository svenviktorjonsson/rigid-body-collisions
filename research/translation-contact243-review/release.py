"""Bounded normal-contact release guesses, full original-law gates."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from research.coulomb_diagnostics import System

def main():
 path=Path('research/hull-translation-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_1.json');d=json.loads(path.read_text());s=System.from_dump(d);tol=d['tolerance_m_s'];native=np.array(d['p']);base=np.array(json.loads(Path(__file__).with_name('positive-warm-FB.json').read_text())['clipped_impulse']);contacts=[c for c in s.contacts if native[c[0]]>0];receipts=[]
 for drop in [39,19]:
  kept=[c for c in contacts if c[0]!=drop];rows=np.array([r for k,t,*_ in kept for r in (k,*t)]);pos={r:i for i,r in enumerate(rows)};cs=tuple((pos[k],tuple(pos[r] for r in t),mu,rn,rt) for k,t,mu,rn,rt in kept);model=System(s.A[np.ix_(rows,rows)],s.b[rows],cs,s.upper[rows]);ns=np.array([c[0] for c in cs]);rn=np.array([c[3] for c in cs]);start=time.perf_counter()
  def fb(p,jac=False):
   F,J=model.equations(p,True);u=p[ns]/rn;w=model.A[ns]@p-model.b[ns];l=np.hypot(u,w);F[ns]=u+w-l
   if jac:
    ca=1-np.divide(u,l,out=np.zeros_like(u),where=l>0);cb=1-np.divide(w,l,out=np.zeros_like(w),where=l>0);J[ns]=cb[:,None]*model.A[ns];J[ns,ns]+=ca/rn;return J
   return F
  r=least_squares(fb,base[rows],jac=lambda p:fb(p,True),max_nfev=1000,ftol=1e-14,xtol=1e-14,gtol=1e-14);p=np.zeros(len(s.b));p[rows]=r.x;ks=np.array([c[0] for c in s.contacts]);p[ks]=np.maximum(p[ks],0);gate=s.gate(p,tol);out=dict(released_normal=drop,remaining_rows=rows.tolist(),nfev=r.nfev,njev=r.njev,elapsed_s=time.perf_counter()-start,gate=gate,impulse=p.tolist());receipts.append(out);print(drop,r.nfev,gate,flush=True)
  Path(__file__).with_name('normal-release.json').write_text(json.dumps(dict(capture=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),attempts=receipts),indent=2)+'\n')
  if gate['accepted']:break
if __name__=='__main__':main()
