"""Retained numerical FB-normal merit exploration; final natural-map gate."""
import json,time,argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from research.coulomb_trust_diagnostic import equations

def main():
 parser=argparse.ArgumentParser();parser.add_argument('capture');parser.add_argument('--steps',type=int,default=10);a=parser.parse_args();d=json.load(open(a.capture));A=np.array(d['A']);b=np.array(d['b']);dep=np.array(d['dependencies']);hi=np.array(d['hi']);ks=np.flatnonzero(dep<0);rn=1/A[ks,ks];p=np.zeros(len(b));hist=[];start=time.perf_counter()
 for alpha in np.linspace(0,1,a.steps+1):
  trial=dict(d);h=hi.copy();h[dep>=0]*=alpha;trial['hi']=h;original,error=equations(trial)
  def f(p,jac=False):
   out=original(p,jac);u=p[ks]/rn;w=A[ks]@p-b[ks];length=np.hypot(u,w)
   if jac:
    ca=1-np.divide(u,length,out=np.zeros_like(u),where=length>0);cb=1-np.divide(w,length,out=np.zeros_like(w),where=length>0)
    out[ks]=cb[:,None]*A[ks];out[ks,ks]+=ca/rn
   else:out[ks]=u+w-length
   return out
  r=least_squares(f,p,jac=lambda p:f(p,True),method='trf',max_nfev=400,ftol=1e-14,xtol=1e-14,gtol=1e-14);p=r.x;hist.append(dict(alpha=alpha,residual=error(p),nfev=r.nfev));print(alpha,error(p),r.nfev,flush=True)
 orig,e=equations(d);w=A@p-b;out=dict(capture=a.capture,method='FB_normal_trf_continuation',steps=a.steps,elapsed_s=time.perf_counter()-start,final_residual=e(p),passive_change_bound_J=float(.5*p@(w-b)),passivity_scale=float(1+np.abs(p*b).sum()),history=hist,p=p.tolist());path=Path(a.capture);name=path.parent.name+'-'+path.stem+'-FB-'+str(a.steps);Path('research/coulomb-trust/'+name+'.json').write_text(json.dumps(out,indent=2));print(name,e(p),flush=True)
if __name__=='__main__':main()
