"""Neutral contact-pressure initialization search; original law accepts outputs."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,least_squares
from research.coulomb_diagnostics import System

def main():
 capture=Path('research/hull-active-completion/results/rejections/fast_shake8_hulls42/reference_1.json');data=json.loads(capture.read_text());sys=System.from_dump(data);tol=data['tolerance_m_s'];p=np.array(json.loads(Path('research/active-direct/39-candidate-more-v2-lapack-native.jsonl').read_text())['p']);velocity=sys.A@p-sys.b;active=[c for c in sys.contacts if p[c[0]]>1e-9 or abs(velocity[c[0]])<tol*10];rows=np.array([r for k,ts,*_ in active for r in (k,*ts)]);_,singular,V=np.linalg.svd(sys.A[:,rows],full_matrices=False);rank=int(np.sum(singular>singular[0]*1e-13));N=np.zeros((len(p),len(rows)-rank));N[rows]=V[rank:].T;ks=np.array([c[0] for c in sys.contacts]);rn=np.array([c[3] for c in sys.contacts]);receipts=[]
 def fb(q,jac=False):
  F,J=sys.equations(q,True);u=q[ks]/rn;w=sys.A[ks]@q-sys.b[ks];length=np.hypot(u,w)
  if jac:
   ca=1-np.divide(u,length,out=np.zeros_like(u),where=length>0);cb=1-np.divide(w,length,out=np.zeros_like(w),where=length>0);J[ks]=cb[:,None]*sys.A[ks];J[ks,ks]+=ca/rn;return J
  F[ks]=u+w-length;return F
 def capacities(z,jac=False):
  q=p+N@z;values=[];grad=[]
  for k,ts,mu,*_ in active:
   t=list(ts);length=np.linalg.norm(q[t]);values.extend((q[k],mu*q[k]-length));grad.extend((N[k],mu*N[k]-(q[t]@N[t])/length if length>0 else mu*N[k]))
  return np.array(grad) if jac else np.array(values)
 for label,k,sign in [('minimum-norm',None,0)]+[(f'pressure-{k}-{name}',k,sign) for k,*_ in active for name,sign in [('minimum',1),('maximum',-1)]]:
  def objective(z):
   q=p+N@z;return .5*q@q if k is None else sign*q[k]
  def gradient(z):return N.T@(p+N@z) if k is None else sign*N[k]
  start=time.perf_counter();r=minimize(objective,np.zeros(N.shape[1]),jac=gradient,constraints=[dict(type='ineq',fun=capacities,jac=lambda z:capacities(z,True))],method='SLSQP',options={'ftol':1e-15,'maxiter':1000});base=p+N@r.x;w=sys.A@base-sys.b;F=sys.equations(base);entry=dict(label=label,canonical_optimizer_success=bool(r.success),canonical_iterations=r.nit,mobility_change_inf=float(np.max(abs(sys.A@(base-p)))),canonical_impulse=base.tolist(),canonical_gate=sys.gate(base,tol),min_capacity=float(np.min(capacities(r.x))),attempts=[]);receipts.append(entry)
  for contact in sorted(active,key=lambda c:np.linalg.norm(F[list(c[1])]),reverse=True):
   kk,ts,mu,*_=contact;t=list(ts);speed=np.linalg.norm(w[t]);initial=base.copy()
   if speed>0:initial[t]=-mu*base[kk]*w[t]/speed
   solve=least_squares(fb,initial,jac=lambda q:fb(q,True),method='trf',max_nfev=300,ftol=1e-14,xtol=1e-14,gtol=1e-14);q=solve.x;gate=sys.gate(q,tol);entry['attempts'].append(dict(opposing_contact=kk,nfev=solve.nfev,initial_impulse=initial.tolist(),impulse=q.tolist(),gate=gate));print(label,kk,gate['accepted'],gate['residual_m_s'],solve.nfev,flush=True)
   dest=Path(__file__).with_name('canonical-probes.json');dest.write_text(json.dumps(dict(capture=str(capture),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),basis_rows=rows.tolist(),basis_rank=rank,null_dimension=N.shape[1],neutral_basis=N.tolist(),attempts=receipts),indent=2)+'\n')
   if gate['accepted']:break
  entry['elapsed_s']=time.perf_counter()-start
if __name__=='__main__':main()
