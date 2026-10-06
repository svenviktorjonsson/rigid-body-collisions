"""Scaled minimax linearized directions, unchanged original-law final gates."""
import ast,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'plan.json').read_text());D=H/'results';D.mkdir(exist_ok=False)
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent','exec'),ns)
assert os.environ['OMP_NUM_THREADS']=='1' and os.environ['OPENBLAS_NUM_THREADS']=='1'
class Limit(Exception):pass
records=[]
for index,(path,sha) in enumerate(plan['inputs'].items()):
 assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha;d=json.loads((ROOT/path).read_text());A=np.array(d['A']);b=np.array(d['b']);p=np.array(d['p']);dep=np.array(d['dependencies']);hi=np.array(d['hi']);tol=d['tolerance_m_s'];seen=set();attempts=[]
 for first in range(len(p)):
  if first in seen:continue
  ids=[first];seen.add(first)
  for k in ids:
   for j in range(len(p)):
    if j not in seen and (A[k,j]!=0 or A[j,k]!=0 or dep[k]==j or dep[j]==k):seen.add(j);ids.append(j)
  inv={k:j for j,k in enumerate(ids)};M=A[np.ix_(ids,ids)];rhs=b[ids];q=p[ids].copy();localdep=np.array([inv[k] if k>=0 else -1 for k in dep[ids]]);upper=hi[ids];local=dict(A=M.tolist(),b=rhs.tolist(),hi=upper.tolist(),dependencies=localdep.tolist(),tolerance_m_s=tol);contacts=[]
  for k in np.flatnonzero(localdep<0):
   ts=np.flatnonzero(localdep==k);contacts.append((int(k),ts,np.linalg.eigvalsh(M[np.ix_(ts,ts)])[-1],upper[ts[0]]))
  check=lambda x:ns['external'](local,{'p':x.tolist(),'w':(M@x-rhs).tolist()})
  if check(q)['accepted']:continue
  if len(q)>128:continue
  n=len(q);raw=0;radius=max(1e-4,float(np.max(abs(q))))
  def equations(x,jac=False):
   global raw
   if raw>=plan['limits']['residual_calls']:raise Limit()
   raw+=1;w=M@x-rhs;F=np.zeros(n);J=np.zeros((n,n))
   for k,ts,eig,mu in contacts:
    zn=x[k]-w[k]/M[k,k];F[k]=(x[k]-max(0,zn))*M[k,k];J[k]=M[k] if zn>0 else np.eye(n)[k]*M[k,k]
    z=x[ts]-w[ts]/eig;length=np.linalg.norm(z);cap=mu*max(0,x[k])
    if length<=cap and cap>0:F[ts]=w[ts];J[ts]=M[ts]
    else:
     u=z/max(length,1e-300);F[ts]=(x[ts]-cap*u)*eig
     B=cap/max(length,1e-300)*(np.eye(2)-np.outer(u,u));J[ts]=(np.eye(n)[ts]-B@(np.eye(n)[ts]-M[ts]/eig))*eig
     if x[k]>=0:J[ts,k]-=mu*u*eig
   return F,J
  try:
   for outer in range(plan['limits']['outer_iterations']):
    F,J=equations(q,True);gate=check(q)
    if gate['accepted']:break
    scale=np.maximum(np.linalg.norm(J,axis=0),1e-12);C=J/scale;f=F/tol;bounds=[]
    for j in range(n):bounds.append((max(-radius*scale[j]/tol,-q[j]*scale[j]/tol) if localdep[j]<0 else -radius*scale[j]/tol,radius*scale[j]/tol))
    bounds.append((0,None));B=np.vstack([np.c_[C,-np.ones(n)],np.c_[-C,-np.ones(n)]]);target=np.r_[-f,f];objective=np.r_[np.zeros(n),1.];lp=linprog(objective,A_ub=B,b_ub=target,bounds=bounds,method='highs');info={'component_rows':ids,'outer':outer,'lp_reported_success':bool(lp.success),'raw_evaluations':raw}
    if not lp.success:attempts.append(info);break
    # A second numerical LP selects a small direction; LP status never accepts physics.
    t=float(lp.x[-1]);secondary_B=np.vstack([np.c_[C,np.zeros((n,n))],np.c_[-C,np.zeros((n,n))],np.c_[np.eye(n),-np.eye(n)],np.c_[-np.eye(n),-np.eye(n)]]);secondary_target=np.r_[np.full(n,t+1e-8)-f,np.full(n,t+1e-8)+f,np.zeros(2*n)];secondary=linprog(np.r_[np.zeros(n),np.ones(n)],A_ub=secondary_B,b_ub=secondary_target,bounds=bounds[:-1]+[(0,None)]*n,method='highs');z=secondary.x[:n] if secondary.success else lp.x[:n];step=tol*z/scale;info.update(predicted_scaled_max=t,original_linearized_residual_max=float(np.max(abs(F+J@step))),linear_lp_slack_scaled=float(np.max(C@z+f-t)),secondary_reported_success=bool(secondary.success))
    score=float(np.max(abs(F)));accepted=False
    for line in range(plan['limits']['line_trials']):
     trial=q+step*2.**(-line);trial[np.flatnonzero(localdep<0)]=np.maximum(0,trial[np.flatnonzero(localdep<0)]);newF,_=equations(trial);newgate=check(trial)
     if newgate['accepted'] or np.max(abs(newF))<score:q=trial;accepted=True;break
    info.update(direction_improved=accepted,candidate_p=q.tolist(),original_gate=check(q));attempts.append(info);save(D/f'{index}-progress.json',attempts)
    if not accepted:radius*=.25
    elif line==0:radius=min(radius*2,max(1.,float(np.max(abs(q)))))
    if radius<1e-16:break
  except Limit:pass
  p[ids]=q
 gate=ns['external'](d,{'p':p.tolist(),'w':(A@p-b).tolist()});candidate=D/f'{index}-candidate.json';save(candidate,dict(d,p=p.tolist()));native=subprocess.run([str(ROOT/'build/spatial/spatial_coulomb_replay'),str(candidate),'0'],capture_output=True,text=True);(D/f'{index}-native.json').write_text(native.stdout);record={'input':path,'accepted':bool(gate['accepted'] and native.returncode==0),'independent':gate,'production_exit':native.returncode,'attempts':attempts};records.append(record);save(D/'progress.json',records);print(index,len(b),'accepted',record['accepted'],'residual',gate['projection_m_s'],flush=True)
save(D/'summary.json',{'complete':True,'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'records':records,'scope':'LPs are numerical directions only. Only unchanged original full law plus production gate accepts; no world/performance claim.'})
