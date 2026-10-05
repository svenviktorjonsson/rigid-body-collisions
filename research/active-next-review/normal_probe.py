"""Normal-only position-capture probes, original full circular/passivity gates."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,least_squares
from scipy.linalg import lstsq
from scipy.sparse.csgraph import connected_components
from research.coulomb_diagnostics import System

def main():
 source=Path('research/hull-active-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_0.json');d=json.loads(source.read_text());sys=System.from_dump(d);ks=np.array([c[0] for c in sys.contacts]);M=sys.A[np.ix_(ks,ks)];b=sys.b[ks];warm=np.array(d['p'])[ks];tol=d['tolerance_m_s'];rn=1/np.diag(M)
 assert all(mu==0 for k,t,mu,*_ in sys.contacts)
 nc,labels=connected_components(M!=0,directed=False);records=[]
 for method in ['SLSQP-warm','SLSQP-cold','FB-warm','natural-warm']:
  initial=np.zeros(len(b)) if 'cold' in method else warm.copy();start=time.perf_counter();history=[]
  def gate(q):
   p=np.zeros(len(sys.b));p[ks]=q;return sys.gate(p,tol)
  if 'SLSQP' in method:
   r=minimize(lambda q:.5*q@M@q-b@q,initial,jac=lambda q:M@q-b,bounds=[(0,None)]*len(b),method='SLSQP',options={'ftol':1e-15,'maxiter':3000},callback=lambda q:history.append(gate(q)))
  else:
   def eq(q,jac=False):
    w=M@q-b
    if 'FB' in method:
     u=q/rn;l=np.hypot(u,w)
     if jac:
      ca=1-np.divide(u,l,out=np.zeros_like(u),where=l>0);cb=1-np.divide(w,l,out=np.zeros_like(w),where=l>0);J=cb[:,None]*M;J[np.diag_indices_from(J)]+=ca/rn;return J
     return u+w-l
    z=q-rn*w
    if jac:
     J=M.copy();off=z<=0;J[off]=0;ii=np.flatnonzero(off);J[ii,ii]=1/rn[ii];return J
    return (q-np.maximum(0,z))/rn
   r=least_squares(eq,initial,jac=lambda q:eq(q,True),method='trf',max_nfev=1000,ftol=1e-14,xtol=1e-14,gtol=1e-14)
  q=r.x;active=np.flatnonzero(q>1e-10);polished=np.zeros(len(q));polished[active]=lstsq(M[np.ix_(active,active)],b[active],cond=1e-13)[0];polished[np.logical_and(polished<0,polished>-1e-12)]=0
  p=np.zeros(len(sys.b));p[ks]=q
  record=dict(method=method,elapsed_s=time.perf_counter()-start,iterations=int(getattr(r,'nit',getattr(r,'nfev',0))),optimizer_success=bool(r.success),message=r.message,gate=gate(q),polished_gate=gate(polished),normal_impulse=q.tolist(),polished_impulse=polished.tolist(),history=history)
  records.append(record);print(method,record['gate'],record['polished_gate'],record['elapsed_s'],flush=True)
  Path(__file__).with_name('normal291-probes.json').write_text(json.dumps(dict(capture=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),normal_rows=ks.tolist(),components=[np.flatnonzero(labels==c).tolist() for c in range(nc)],min_eigenvalue=float(np.linalg.eigvalsh(M)[0]),attempts=records),indent=2)+'\n')
if __name__=='__main__':main()
