"""Independent exact-law full/reduced search of a frozen new hull capture."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from research.coulomb_diagnostics import System

def main():
 a=argparse.ArgumentParser();a.add_argument('--scope',choices=['full','positive'],default='positive');a.add_argument('--start',choices=['warm','cold'],default='warm');a.add_argument('--merit',choices=['FB','natural'],default='FB');a.add_argument('--max',type=int,default=1000);args=a.parse_args()
 path=Path('research/hull-translation-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_1.json');d=json.loads(path.read_text());s=System.from_dump(d);tol=d['tolerance_m_s'];warm=np.array(d['p']);ks=np.array([c[0] for c in s.contacts]);contacts=s.contacts if args.scope=='full' else [c for c in s.contacts if warm[c[0]]>0];rows=np.array([r for k,t,*_ in contacts for r in (k,*t)]);position={r:i for i,r in enumerate(rows)};map_contacts=tuple((position[k],tuple(position[r] for r in t),mu,rn,rt) for k,t,mu,rn,rt in contacts);model=System(s.A[np.ix_(rows,rows)],s.b[rows],map_contacts,s.upper[rows]);ns=np.array([c[0] for c in model.contacts]);rn=np.array([c[3] for c in model.contacts]);initial=warm[rows] if args.start=='warm' else np.zeros(len(rows));history=[]
 def eq(p,jac=False):
  F,J=model.equations(p,True)
  if args.merit=='FB':
   u=p[ns]/rn;w=model.A[ns]@p-model.b[ns];length=np.hypot(u,w);F[ns]=u+w-length
   if jac:
    ca=1-np.divide(u,length,out=np.zeros_like(u),where=length>0);cb=1-np.divide(w,length,out=np.zeros_like(w),where=length>0);J[ns]=cb[:,None]*model.A[ns];J[ns,ns]+=ca/rn
  if not jac:history.append(float(model.residual(p)))
  return J if jac else F
 start=time.perf_counter();r=least_squares(eq,initial,jac=lambda p:eq(p,True),method='trf',max_nfev=args.max,ftol=1e-14,xtol=1e-14,gtol=1e-14);p=np.zeros(len(s.b));p[rows]=r.x;clipped=p.copy();clipped[ks]=np.maximum(0,clipped[ks]);F=s.equations(clipped);w=s.A@clipped-s.b;worst=sorted([dict(normal=k,pn=float(clipped[k]),wn=float(w[k]),residual_m_s=float(max(abs(F[k]),np.linalg.norm(F[list(t)])))) for k,t,*_ in s.contacts],key=lambda r:-r['residual_m_s'])[:10]
 out=dict(capture=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope=args.scope,start=args.start,merit=args.merit,rows=rows.tolist(),nfev=r.nfev,njev=r.njev,max_nfev=args.max,optimizer_success=bool(r.success),message=r.message,elapsed_s=time.perf_counter()-start,gate=s.gate(p,tol),clipped_gate=s.gate(clipped,tol),impulse=p.tolist(),clipped_impulse=clipped.tolist(),residual_history=history,worst_contacts=worst)
 dest=Path(__file__).with_name(args.scope+'-'+args.start+'-'+args.merit+'.json');assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['rows','impulse','clipped_impulse','residual_history','worst_contacts']},indent=2),flush=True)
if __name__=='__main__':main()
