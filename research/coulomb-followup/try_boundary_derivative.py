import json,numpy as np
from research.coulomb_diagnostics import System,solve,gauge_candidates
class BoundarySystem:
 def __init__(self,system,epsilon):self.__dict__.update(system.__dict__);self.original=system;self.epsilon=epsilon
 def equations(self,p,jacobian=False):
  if not jacobian:return self.original.equations(p)
  F,J=self.original.equations(p,True);w=self.A@p-self.b
  for k,ts,mu,rn,rt in self.contacts:
   t=list(ts);z=p[t]-rt*w[t];length=np.linalg.norm(z);cap=mu*max(0.,p[k])
   if cap>0 and length>0 and length<=cap and cap-length<=self.epsilon*max(1.,cap):
    d=z/length;D=cap/length*(np.eye(2)-np.outer(d,d));J[t]=D@self.A[t];J[np.ix_(t,t)]+=(np.eye(2)-D)/rt;J[t,k]-=mu*d/rt
  return F,J
 def gate(self,*args):return self.original.gate(*args)
D=json.load(open('research/coulomb-followup/hull42-second-rejected.json'));S=System.from_dump(D);p=solve(S,D['p'],max_nfev=0)['impulse'];starts=[('warm',np.asarray(D['p'])),('stalled',p)]+[('gauge'+str(i),c['impulse']) for i,c in enumerate(gauge_candidates(S,p))];records=[]
for epsilon in [0.,1e-12,1e-10,1e-8,1e-6]:
 C=BoundarySystem(S,epsilon)
 for name,q in starts:
  r=solve(C,q,max_newton=64,max_nfev=0);records.append(dict(epsilon=epsilon,start=name,accepted=r['accepted'],residual=r['residual_m_s'],iterations=len(r['jacobian_ranks'])))
print('accepted',[r for r in records if r['accepted']]);print('best',min(records,key=lambda r:r['residual']))
json.dump(records,open('research/coulomb-followup/hull42-boundary-derivative.json','w'),indent=2)
