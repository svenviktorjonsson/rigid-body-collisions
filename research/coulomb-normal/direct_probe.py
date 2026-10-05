"""Fresh captures: exact-map warm/cold/FB numerical probes, retained receipts."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from research.coulomb_trust_diagnostic import equations

def main():
 a=argparse.ArgumentParser();a.add_argument('capture');a.add_argument('--merit',choices=['projection','FB'],default='projection');a.add_argument('--start',choices=['warm','cold'],default='warm');a.add_argument('--max',type=int,default=500);args=a.parse_args()
 path=Path(args.capture);d=json.loads(path.read_text());A=np.array(d['A']);b=np.array(d['b']);ks=np.flatnonzero(np.array(d['dependencies'])<0);rn=1/A[ks,ks];original,error=equations(d)
 def f(p,jac=False):
  out=original(p,jac)
  if args.merit=='FB':
   u=p[ks]/rn;w=A[ks]@p-b[ks];length=np.hypot(u,w)
   if jac:
    ca=1-np.divide(u,length,out=np.zeros_like(u),where=length>0);cb=1-np.divide(w,length,out=np.zeros_like(w),where=length>0);out[ks]=cb[:,None]*A[ks];out[ks,ks]+=ca/rn
   else:out[ks]=u+w-length
  return out
 p=np.array(d['p']) if args.start=='warm' else np.zeros(len(b));start=time.perf_counter();result=least_squares(f,p,jac=lambda p:f(p,True),method='trf',max_nfev=args.max,ftol=1e-14,xtol=1e-14,gtol=1e-14);p=result.x;w=A@p-b
 out=dict(capture=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),merit=args.merit,start=args.start,elapsed_s=time.perf_counter()-start,original_residual=error(p),nfev=result.nfev,njev=result.njev,message=result.message,cost=result.cost,energy_J=float(.5*p@(w-b)),p=p.tolist());dest=Path('research/coulomb-normal')/(path.parent.name+'-'+path.stem+'-'+args.merit+'-'+args.start+'.json');assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='p'},flush=True)
if __name__=='__main__':main()
