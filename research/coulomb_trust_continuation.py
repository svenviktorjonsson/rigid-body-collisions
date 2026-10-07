"""Exploratory numerical friction homotopy; only final physical law may pass."""
import json,time,hashlib,argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from research.coulomb_trust_diagnostic import equations

def main():
 parser=argparse.ArgumentParser();parser.add_argument('capture');parser.add_argument('--steps',type=int,default=40);parser.add_argument('--method',default='trf');args=parser.parse_args()
 path=Path(args.capture);d=json.loads(path.read_text());target_hi=np.asarray(d['hi']);dep=np.asarray(d['dependencies']);p=np.zeros(len(dep));history=[];start=time.perf_counter()
 for alpha in np.linspace(0,1,args.steps+1):
  trial=dict(d);hi=target_hi.copy();hi[dep>=0]*=alpha;trial['hi']=hi.tolist();f,e=equations(trial)
  result=least_squares(f,p,jac=lambda x:f(x,True),method=args.method,max_nfev=400,ftol=1e-14,xtol=1e-14,gtol=1e-14)
  p=result.x;history.append(dict(alpha=float(alpha),residual=float(e(p)),nfev=result.nfev,optimality=result.optimality,message=result.message))
  print(alpha,e(p),result.nfev,flush=True)
 f,e=equations(d);A=np.asarray(d['A']);b=np.asarray(d['b']);w=A@p-b
 out=dict(capture=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),method=args.method,continuation_steps=args.steps,
          elapsed_s=time.perf_counter()-start,final_residual=float(e(p)),passive_change_bound_J=float(.5*p@(w-b)),passivity_scale=float(1+np.abs(p*b).sum()),history=history,p=p.tolist())
 name=path.parent.name+'-'+path.stem+'-homotopy-'+str(args.steps)+'-'+args.method
 Path('research/coulomb-trust/'+name+'.json').write_text(json.dumps(out,indent=2)+'\n');print(name,out['final_residual'],out['elapsed_s'],flush=True)
if __name__=='__main__':main()
