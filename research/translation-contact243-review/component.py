"""Exact positive-support mobility components, original all-row acceptance."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.csgraph import connected_components
from research.coulomb_diagnostics import System

def main():
 cap=Path('research/hull-translation-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_1.json');d=json.loads(cap.read_text());s=System.from_dump(d);p=np.array(d['p']);tol=d['tolerance_m_s'];contacts=[c for c in s.contacts if p[c[0]]>0];rows=np.array([r for k,t,*_ in contacts for r in [k,*t]]);adj=s.A[np.ix_(rows,rows)]!=0;position={r:i for i,r in enumerate(rows)}
 for k,t,*_ in contacts:
  i=[position[r] for r in [k,*t]];adj[np.ix_(i,i)]=True
 nc,labels=connected_components(adj,directed=False);F=s.equations(p);ranking=sorted(range(nc),key=lambda c:-max(abs(F[rows[labels==c]])));receipts=[]
 for component in ranking:
  selected=rows[labels==component];pos={r:i for i,r in enumerate(selected)};cs=tuple((pos[k],tuple(pos[r] for r in t),mu,rn,rt) for k,t,mu,rn,rt in contacts if k in pos);model=System(s.A[np.ix_(selected,selected)],s.b[selected],cs,s.upper[selected]);ns=np.array([c[0] for c in cs]);rn=np.array([c[3] for c in cs]);initial=p.copy();start=time.perf_counter()
  def fb(q,jac=False):
   FF,J=model.equations(q,True);u=q[ns]/rn;w=model.A[ns]@q-model.b[ns];l=np.hypot(u,w);FF[ns]=u+w-l
   if jac:
    ca=1-np.divide(u,l,out=np.zeros_like(u),where=l>0);cb=1-np.divide(w,l,out=np.zeros_like(w),where=l>0);J[ns]=cb[:,None]*model.A[ns];J[ns,ns]+=ca/rn;return J
   return FF
  r=least_squares(fb,p[selected],jac=lambda q:fb(q,True),max_nfev=1000,ftol=1e-14,xtol=1e-14,gtol=1e-14);p[selected]=r.x;ks=np.array([c[0] for c in s.contacts]);p[ks]=np.maximum(p[ks],0);gate=s.gate(p,tol);record=dict(component=component,rows=selected.tolist(),nfev=r.nfev,njev=r.njev,max_nfev=1000,elapsed_s=time.perf_counter()-start,initial_impulse=initial.tolist(),impulse=p.tolist(),gate=gate);receipts.append(record);print(len(selected),r.nfev,r.njev,gate,flush=True)
  Path(__file__).with_name('component-warm-FB.json').write_text(json.dumps(dict(capture=str(cap),capture_sha256=hashlib.sha256(cap.read_bytes()).hexdigest(),positive_components=[rows[labels==c].tolist() for c in range(nc)],attempts=receipts),indent=2)+'\n')
  if gate['accepted']:break
if __name__=='__main__':main()
