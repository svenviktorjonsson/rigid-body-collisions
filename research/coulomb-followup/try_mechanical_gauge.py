import json,numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize
from research.coulomb_diagnostics import System,solve
D=json.load(open('research/coulomb-followup/hull42-second-rejected.json'));S=System.from_dump(D);rr=solve(S,D['p'],max_nfev=0);p=rr['impulse'];w=S.A@p-S.b;B=null_space(S.A,rcond=1e-12)
C=np.vstack([B[[k,*t]] for k,t,*_ in S.contacts if w[k]>1e-8]);B=B@null_space(C,rcond=1e-12)
def cons(c):
 q=p+B@c
 return np.array([x for k,t,mu,*_ in S.contacts if w[k]<=1e-8 for x in [q[k],mu*q[k]-np.linalg.norm(q[list(t)])]])
def jac(c):
 q=p+B@c;rows=[]
 for k,t,mu,*_ in S.contacts:
  if w[k]>1e-8:continue
  t=list(t);r=np.linalg.norm(q[t]);rows.extend([B[k],mu*B[k]-(q[t]@B[t]/r if r>1e-15 else 0)])
 return np.asarray(rows)
records=[]
for k in [0,1,2,4,7,8,13,14]:
 t=list(S.contacts[k][1]);mu=S.contacts[k][2];direction=w[t]/np.linalg.norm(w[t]);E=B[t]+mu*np.outer(direction,B[k]);rhs=-p[t]-mu*p[k]*direction
 result=minimize(lambda c:.5*c@c,np.linalg.lstsq(E,rhs,rcond=1e-12)[0],jac=lambda c:c,method='SLSQP',constraints=[{'type':'ineq','fun':cons,'jac':jac},{'type':'eq','fun':lambda c:E@c-rhs,'jac':lambda c:E}],options={'ftol':1e-14,'maxiter':500})
 q=p+B@result.x;fit=np.linalg.norm(E@result.x-rhs);margin=min(cons(result.x));change=max(abs(S.A@(q-p)))
 r=solve(S,q,max_nfev=0,max_newton=128) if fit<1e-10 and margin>-1e-10 else None
 rec=dict(contact=k,success=result.success,message=result.message,fit=fit,margin=margin,velocity_change=change,initial_residual=S.residual(q),accepted=r and r['accepted'],residual=r and r['residual_m_s'],impulse=q.tolist(),solved_impulse=r['impulse'].tolist() if r else None)
 records.append(rec);print({key:value for key,value in rec.items() if key not in ['impulse','solved_impulse']})
json.dump(records,open('research/coulomb-followup/hull42-mechanical-gauge.json','w'),indent=2)
