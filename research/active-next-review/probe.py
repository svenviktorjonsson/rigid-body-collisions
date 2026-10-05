"""Independent frozen next-capture probes; unchanged original-law final gate."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from research.coulomb_diagnostics import System,analyze

def main():
 a=argparse.ArgumentParser();a.add_argument('capture');a.add_argument('--start',choices=('warm','cold'),default='warm');a.add_argument('--merit',choices=('natural','FB'),default='natural');a.add_argument('--max',type=int,default=1000);args=a.parse_args()
 path=Path(args.capture);data=json.loads(path.read_text());sys=System.from_dump(data);tol=data['tolerance_m_s'];ks=np.array([c[0] for c in sys.contacts]);rn=np.array([c[3] for c in sys.contacts]);p=np.array(data['p']) if args.start=='warm' else np.zeros(len(sys.b));initial=p.copy();history=[]
 def eq(p,jac=False):
  out=sys.equations(p,jac)
  if jac: F,J=out
  else: F=out
  if args.merit=='FB':
   u=p[ks]/rn;w=sys.A[ks]@p-sys.b[ks];length=np.hypot(u,w)
   F[ks]=u+w-length
   if jac:
    ca=1-np.divide(u,length,out=np.zeros_like(u),where=length>0);cb=1-np.divide(w,length,out=np.zeros_like(w),where=length>0)
    J[ks]=cb[:,None]*sys.A[ks];J[ks,ks]+=ca/rn
  if not jac:history.append(dict(residual_m_s=float(sys.residual(p)),merit=float(F@F)))
  return J if jac else F
 start=time.perf_counter();r=least_squares(eq,p,jac=lambda p:eq(p,True),method='trf',max_nfev=args.max,ftol=1e-14,xtol=1e-14,gtol=1e-14);p=r.x
 report=dict(capture=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),method=args.merit,start=args.start,max_nfev=args.max,nfev=r.nfev,njev=r.njev,elapsed_s=time.perf_counter()-start,optimizer_success=bool(r.success),message=r.message,initial=analyze(sys,initial,tol),final=analyze(sys,p,tol),gate=sys.gate(p,tol),impulse=p.tolist(),history=history)
 dest=Path(__file__).parent/(path.parent.name+'-'+path.stem+'-'+args.merit+'-'+args.start+'.json');assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['capture','method','start','nfev','elapsed_s','gate']},indent=2),flush=True)
if __name__=='__main__':main()
