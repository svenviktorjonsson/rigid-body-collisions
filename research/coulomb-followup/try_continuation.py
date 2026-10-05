import json,numpy as np
from research.coulomb_diagnostics import System,solve
D=json.load(open('research/coulomb-followup/hull42-second-rejected.json'));S=System.from_dump(D)
records=[]
for count in [10,50,200]:
 p=np.zeros(len(S.b));path=[]
 for i in range(count+1):
  t=i/count;C=System(S.A,S.b,tuple((k,ts,mu*t,rn,rt) for k,ts,mu,rn,rt in S.contacts),S.upper)
  r=solve(C,p,max_newton=128,max_nfev=0);p=r['impulse'];path.append(dict(fraction=t,residual=r['residual_m_s'],accepted=r['accepted']))
 r=solve(S,p,max_newton=128,max_nfev=3000);records.append(dict(steps=count,accepted=r['accepted'],residual=r['residual_m_s'],path=path,impulse=r['impulse'].tolist()));print(count,r['accepted'],r['residual_m_s'],sum(x['accepted'] for x in path))
json.dump(records,open('research/coulomb-followup/hull42-continuation.json','w'),indent=2)
