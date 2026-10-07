"""Exact-law numerical search using De Saxce's corrected Coulomb cone map.

The corrected map is a search merit only. Original normal/disk projection and
finite passivity checks accept a candidate; no compliance/material changes.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from research.coulomb_trust_diagnostic import equations

def cone_map(data):
 A=np.asarray(data['A']);b=np.asarray(data['b']);dep=np.asarray(data['dependencies']);hi=np.asarray(data['hi']);n=len(b);contacts=[]
 for k in np.flatnonzero(dep<0):
  ts=np.flatnonzero(dep==k);assert len(ts)==2
  rows=np.r_[k,ts];mu=hi[ts[0]];assert hi[ts[1]]==mu and mu>=0
  rho=1/np.linalg.eigvalsh(A[np.ix_(rows,rows)])[-1];contacts.append((rows,mu,rho))
 def calculate(p,jac=False):
  w=A@p-b;F=np.zeros(n);J=np.zeros((n,n)) if jac else None
  for rows,mu,rho in contacts:
   velocities=w[rows];slip=np.linalg.norm(velocities[1:]);u=velocities.copy();u[0]+=mu*slip;y=p[rows]-rho*u;a=y[0];t=y[1:];r=np.linalg.norm(t)
   if a>=0 and r<=mu*a:proj=y;D=np.eye(3)
   elif a+mu*r<=0:proj=np.zeros(3);D=np.zeros((3,3))
   else:
    # Euclidean projection onto ||pt||<=mu*pn in physical impulse units.
    d=1+mu*mu;pn=(a+mu*r)/d;direction=t/r if r>0 else np.zeros(2);proj=np.r_[pn,mu*pn*direction]
    D=np.zeros((3,3));D[0,0]=1/d;D[0,1:]=D[1:,0]=mu/d*direction
    if r>0:D[1:,1:]=mu*pn/r*(np.eye(2)-np.outer(direction,direction))+mu*mu/d*np.outer(direction,direction)
   F[rows]=(p[rows]-proj)/rho
   if jac:
    Du=A[rows].copy()
    if slip>0:Du[0]+=mu*(velocities[1:]/slip)@A[rows[1:]]
    E=np.eye(n)[rows];J[rows]=(E-D@(E-rho*Du))/rho
  return J if jac else F
 return calculate

def main():
 parser=argparse.ArgumentParser();parser.add_argument('capture');parser.add_argument('--start',choices=['warm','cold'],default='warm');parser.add_argument('--max',type=int,default=1000);a=parser.parse_args();path=Path(a.capture);d=json.loads(path.read_text());f=cone_map(d);original,error=equations(d);p=np.asarray(d['p']) if a.start=='warm' else np.zeros(len(d['b']));start=time.perf_counter();result=least_squares(f,p,jac=lambda p:f(p,True),method='trf',max_nfev=a.max,ftol=1e-14,xtol=1e-14,gtol=1e-14);p=result.x;A=np.asarray(d['A']);b=np.asarray(d['b']);dep=np.asarray(d['dependencies']);ks=np.flatnonzero(dep<0);tol=d['tolerance_m_s'];valid=np.all(p[ks]>=-tol/A[ks,ks]);p[ks]=np.maximum(0,p[ks]);w=A@p-b;energy=float(.5*p@(w-b));scale=float(1+np.abs(p*b).sum());physical_error=float(error(p));finite=np.all(np.isfinite(p)) and np.all(np.isfinite(w)) and np.isfinite(energy) and np.isfinite(scale);bounds=np.all(p[ks]<=np.asarray(d['hi'])[ks]);accepted=bool(valid and finite and bounds and physical_error<=tol and energy<=tol*scale)
 out=dict(capture=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),map='De_Saxce_corrected_Euclidean_K_mu_projection',start=a.start,elapsed_s=time.perf_counter()-start,nfev=result.nfev,njev=result.njev,message=result.message,original_projection_residual_m_s=physical_error,original_tolerance_m_s=tol,passive_energy_bound_J=energy,passivity_scale=scale,accepted_by_original_gate=accepted,p=p.tolist());directory=Path('research/coulomb-cone');directory.mkdir(exist_ok=True);dest=directory/(path.parent.name+'-'+path.stem+'-'+a.start+'.json');assert not dest.exists(),'Retain earlier numerical probes';dest.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='p'},flush=True)
if __name__=='__main__':main()
