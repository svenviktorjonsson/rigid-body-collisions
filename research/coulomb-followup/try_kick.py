import json,numpy as np
from research.coulomb_diagnostics import System,solve
D=json.load(open('research/coulomb-followup/hull42-second-rejected.json'));S=System.from_dump(D);r=solve(S,D['p'],max_nfev=0);p=r['impulse'];w=S.A@p-S.b;records=[]
for k in [0,1,2]:
 t=list(S.contacts[k][1]);mu=S.contacts[k][2]
 for angle in np.linspace(0,2*np.pi,16,endpoint=False):
  q=p.copy();q[t]=mu*q[k]*np.array([np.cos(angle),np.sin(angle)])
  rr=solve(S,q,max_newton=128,max_nfev=0)
  record=dict(contact=k,angle=float(angle),accepted=rr['accepted'],residual=rr['residual_m_s'],impulse=rr['impulse'].tolist());records.append(record)
  if rr['accepted']:print('SUCCESS',k,angle,rr['residual_m_s']);break
 else:continue
 break
else:print('best',min(r['residual'] for r in records))
json.dump(records,open('research/coulomb-followup/hull42-kick.json','w'),indent=2)
