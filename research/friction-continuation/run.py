"""Bounded isolated friction homotopy; final original-law gate mandatory."""
import ast,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'plan.json').read_text());D=H/'results-v2';D.mkdir(exist_ok=False)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ['OMP_NUM_THREADS']=='1' and os.environ['OPENBLAS_NUM_THREADS']=='1'
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent', 'exec'),ns)
paths=[Path(__file__),H/'plan.json',ROOT/'build/spatial/spatial_coulomb_replay'];guards={str(p):sha(p) for p in paths};save(D/'provenance.json',{'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'guards':guards});records=[]
class Limit(Exception):pass
for index,(path,digest) in enumerate(plan['inputs'].items()):
 assert sha(ROOT/path)==digest;data=json.loads((ROOT/path).read_text());A=np.array(data['A']);b=np.array(data['b']);p=np.array(data['p']);dep=np.array(data['dependencies']);upper=np.array(data['hi']);tol=data['tolerance_m_s'];seen=set();attempts=[]
 for first in range(len(p)):
  if first in seen:continue
  ids=[first];seen.add(first)
  for k in ids:
   for j in range(len(p)):
    if j not in seen and (A[k,j]!=0 or A[j,k]!=0 or dep[k]==j or dep[j]==k):seen.add(j);ids.append(j)
  inv={k:j for j,k in enumerate(ids)};M=A[np.ix_(ids,ids)];rhs=b[ids];q=p[ids].copy();localdep=np.array([inv[k] if k>=0 else -1 for k in dep[ids]]);hi=upper[ids];local=dict(A=M.tolist(),b=rhs.tolist(),hi=hi.tolist(),dependencies=localdep.tolist(),tolerance_m_s=tol);check=lambda x:ns['external'](local,{'p':x.tolist(),'w':(M@x-rhs).tolist()})
  if check(q)['accepted']:continue
  if len(q)>plan['limits']['component_rows']:attempts.append({'component_rows':ids,'declined':'component_cap'});continue
  n=len(q);I=np.eye(n);contacts=[]
  for k in np.flatnonzero(localdep<0):
   ts=np.flatnonzero(localdep==k);contacts.append((k,ts,np.linalg.eigvalsh(M[np.ix_(ts,ts)])[-1],hi[ts[0]]))
  fraction=0.;step=.125;accepted_fraction=None
  for stage in range(plan['limits']['maximum_stage_attempts']):
   target=0. if accepted_fraction is None else min(1.,accepted_fraction+step);box={'raw':0,'best':q.copy(),'score':float('inf')};start=time.perf_counter()
   def equations(x,jac=False):
    if not jac:
     if box['raw']>=plan['limits']['raw_evaluations_per_stage']:raise Limit()
     box['raw']+=1
    w=M@x-rhs;F=np.zeros(n);J=np.zeros((n,n))
    for k,ts,eig,mu in contacts:
     zn=x[k]-w[k]/M[k,k];F[k]=(x[k]-max(0,zn))*M[k,k];J[k]=M[k] if zn>0 else I[k]*M[k,k]
     z=x[ts]-w[ts]/eig;length=np.linalg.norm(z);cap=target*mu*max(0,x[k])
     if length<=cap and cap>0:F[ts]=w[ts];J[ts]=M[ts]
     else:
      u=z/max(length,1e-300);F[ts]=(x[ts]-cap*u)*eig;B=cap/max(length,1e-300)*(np.eye(len(ts))-np.outer(u,u));J[ts]=(I[ts]-B@(I[ts]-M[ts]/eig))*eig
      if x[k]>=0:J[ts,k]-=target*mu*u*eig
    if not jac:
     score=float(np.max(abs(F)))
     if score<box['score']:box['score']=score;box['best']=x.copy()
    return J if jac else F
   try:least_squares(equations,q,jac=lambda x:equations(x,True),method='trf',x_scale='jac',ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=1024)
   except Limit:pass
   candidate=box['best'].copy();stage_hi=hi.copy();stage_hi[localdep>=0]*=target
   for k,ts,eig,mu in contacts:
    if -tol/M[k,k]<=candidate[k]<0:candidate[k]=0
    cap=target*mu*max(0,candidate[k]);length=np.linalg.norm(candidate[ts]);candidate[ts]*=min(1.,cap/max(length,1e-300))
   stage_local=dict(local,hi=stage_hi.tolist());gate=ns['external'](stage_local,{'p':candidate.tolist(),'w':(M@candidate-rhs).tolist()});passed=bool(gate['accepted']);attempts.append({'component_rows':ids,'stage':stage,'friction_fraction':target,'accepted_intermediate':passed,'raw_evaluations':box['raw'],'candidate_p':candidate.tolist(),'independent':gate,'elapsed_s_descriptive':time.perf_counter()-start});save(D/f'{index}-progress.json',attempts)
   print(index,len(q),stage,target,passed,gate['projection_m_s'],flush=True)
   if passed:
    q=candidate;accepted_fraction=target
    if target==1:break
    step=min(.25,step*1.5)
   else:
    if accepted_fraction is None:break
    step*=.5
    if step<plan['limits']['minimum_friction_fraction_step']:break
  if accepted_fraction==1 and check(q)['accepted']:p[ids]=q
 gate=ns['external'](data,{'p':p.tolist(),'w':(A@p-b).tolist()});capture=D/f'{index}-candidate.json';save(capture,dict(data,p=p.tolist()));native=subprocess.run([str(ROOT/'build/spatial/spatial_coulomb_replay'),str(capture),'0'],capture_output=True,text=True);(D/f'{index}-native.stdout.json').write_text(native.stdout);(D/f'{index}-native.stderr').write_text(native.stderr);record={'input':path,'accepted':bool(gate['accepted'] and native.returncode==0),'independent':gate,'native_exit':native.returncode,'attempts':attempts};records.append(record);save(D/'progress.json',records)
assert all(sha(p)==digest for p,digest in guards.items());save(D/'summary.json',{'complete':True,'guards_unchanged':True,'records':records,'scope':plan['rule']})
