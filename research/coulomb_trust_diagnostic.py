"""Exploratory trust-region solving of exact archived circular contact maps."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares


def equations(data):
    A=np.asarray(data['A']);b=np.asarray(data['b']);dep=np.asarray(data['dependencies']);hi=np.asarray(data['hi']);n=len(b)
    contacts=[]
    for k in np.flatnonzero(dep<0):
        t,s=np.flatnonzero(dep==k);rt=1/np.linalg.eigvalsh(A[np.ix_([t,s],[t,s])])[-1]
        contacts.append((k,t,s,hi[t],1/A[k,k],rt))
    def calc(p,jac=False):
        w=A@p-b;F=np.zeros(n);J=np.zeros((n,n)) if jac else None
        for k,t,s,mu,rn,rt in contacts:
            zn=p[k]-rn*w[k];F[k]=(p[k]-max(0.,zn))/rn
            if jac:J[k]=A[k] if zn>0 else np.eye(n)[k]/rn
            rows=np.array([t,s]);z=p[rows]-rt*w[rows];length=np.linalg.norm(z);cap=mu*max(0.,p[k])
            if length<=cap and cap>0:
                F[rows]=w[rows]
                if jac:J[rows]=A[rows]
            else:
                direction=z/length if length>0 else np.zeros(2);F[rows]=(p[rows]-cap*direction)/rt
                if jac:
                    D=cap/length*(np.eye(2)-np.outer(direction,direction)) if length>0 else np.zeros((2,2))
                    E=np.eye(n)[rows];J[rows]=(E-D@(E-rt*A[rows]))/rt
                    if p[k]>0:J[rows,k]-=mu*direction/rt
        return J if jac else F
    def error(p):
        F=calc(p);return max(max(abs(F[k]),np.hypot(F[t],F[s])) for k,t,s,*_ in contacts)
    return calc,error

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('capture');parser.add_argument('--method',default='trf');parser.add_argument('--start',default='warm');parser.add_argument('--max',type=int,default=1000)
    args=parser.parse_args();path=Path(args.capture);d=json.loads(path.read_text());f,e=equations(d);start=np.array(d['p']) if args.start=='warm' else np.zeros(len(d['b']));t=time.perf_counter()
    result=least_squares(f,start,jac=lambda p:f(p,True),method=args.method,max_nfev=args.max,
                         ftol=1e-14,xtol=1e-14,gtol=1e-14)
    p=result.x;A=np.asarray(d['A']);b=np.asarray(d['b']);w=A@p-b
    out=dict(capture=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),method=args.method,start=args.start,
             elapsed_s=time.perf_counter()-t,initial_residual=e(start),final_residual=e(p),cost=result.cost,
             optimality=result.optimality,message=result.message,nfev=result.nfev,njev=result.njev,
             passive_change_bound_J=float(.5*p@(w-b)),passivity_scale=float(1+np.abs(p*b).sum()),p=p.tolist())
    name=path.parent.name+'-'+path.stem+'-'+args.method+'-'+args.start
    Path('research/coulomb-trust/'+name+'.json').write_text(json.dumps(out,indent=2)+'\n')
    print(name,{k:v for k,v in out.items() if k!='p'},flush=True)
if __name__=='__main__':main()
