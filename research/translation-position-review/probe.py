"""Strict normal-only matrix probes; explicit trial gates and failures."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,least_squares

def main():
 cap=Path('research/translation-position-diagnostic/results/rejected-normal-system.json');d=json.loads(cap.read_text());A=np.array(d['A']);b=np.array(d['b']);upper=np.array(d['hi']);seed=np.array(d['p']);tol=d['tolerance_m_s'];rn=1/np.diag(A);attempts=[]
 assert all(v==-1 for v in d['dependencies'])
 def gate(p):
  w=A@p-b;res=np.max(abs(p-np.maximum(0,p-rn*w))/rn);energy=.5*p@A@p-b@p;scale=1+np.sum(abs(p*b));return dict(accepted=bool(np.isfinite(p).all() and np.isfinite(w).all() and np.isfinite(energy) and np.isfinite(scale) and np.min(p)>=0 and np.all(p<=upper) and res<=tol and energy<=tol*scale),residual_m_s=float(res),passive_change_bound_J=float(energy),passivity_scale=float(scale),normal_min=float(np.min(p)),pressure_max=float(np.max(abs(p))))
 for method in ['SLSQP-warm','scaled-SLSQP-warm','FB-warm','FB-cold']:
  p=seed.copy() if 'warm' in method else np.zeros(len(seed));start=time.perf_counter();history=[]
  if 'SLSQP' in method:
   factor=1/max(np.max(b*b/np.diag(A)),1e-30) if 'scaled' in method else 1
   r=minimize(lambda p:factor*(.5*p@A@p-b@p),p,jac=lambda p:factor*(A@p-b),bounds=[(0,None)]*len(b),method='SLSQP',options={'ftol':1e-15,'maxiter':3000},callback=lambda p:history.append(gate(p)))
  else:
   def fb(p,jac=False):
    u=p/rn;w=A@p-b;l=np.hypot(u,w)
    if jac:
     ca=1-np.divide(u,l,out=np.zeros_like(u),where=l>0);cb=1-np.divide(w,l,out=np.zeros_like(w),where=l>0);J=cb[:,None]*A;J[np.diag_indices_from(J)]+=ca/rn;return J
    return u+w-l
   r=least_squares(fb,p,jac=lambda p:fb(p,True),max_nfev=3000,ftol=1e-14,xtol=1e-14,gtol=1e-14)
  q=r.x.copy();q=np.maximum(q,0);record=dict(method=method,nfev=int(getattr(r,'nfev',0)),iterations=int(getattr(r,'nit',0)),optimizer_success=bool(r.success),message=r.message,elapsed_s=time.perf_counter()-start,initial_impulse=p.tolist(),raw_impulse=r.x.tolist(),impulse=q.tolist(),gate=gate(q),history=history);attempts.append(record);print(method,record['gate'],record['elapsed_s'],flush=True)
  Path(__file__).with_name('normal-probes.json').write_text(json.dumps(dict(capture=str(cap),capture_sha256=hashlib.sha256(cap.read_bytes()).hexdigest(),attempts=attempts),indent=2)+'\n')
if __name__=='__main__':main()
