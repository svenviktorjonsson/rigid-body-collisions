"""Eliminate original normal/sticking equations; solve only sliding angles."""
import ast,hashlib,itertools,json,os,subprocess
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'plan.json').read_text());D=H/'results';D.mkdir(exist_ok=False)
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent','exec'),ns)
assert os.environ['OMP_NUM_THREADS']=='1' and os.environ['OPENBLAS_NUM_THREADS']=='1'
path,sha=next(iter(plan['inputs'].items()));assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
d=json.loads((ROOT/path).read_text());A=np.array(d['A']);b=np.array(d['b']);p=np.array(d['p']);ids=plan['component_rows'];M=A[np.ix_(ids,ids)];rhs=b[ids];warm=p[ids].copy();inv={k:i for i,k in enumerate(ids)};dep=np.array([inv[k] if k>=0 else -1 for k in np.array(d['dependencies'])[ids]]);upper=np.array(d['hi'])[ids];local=dict(A=M.tolist(),b=rhs.tolist(),dependencies=dep.tolist(),hi=upper.tolist(),tolerance_m_s=d['tolerance_m_s']);contacts=[(int(k),np.flatnonzero(dep==k),upper[np.flatnonzero(dep==k)[0]]) for k in np.flatnonzero(dep<0)];active=[j for j,(k,ts,mu) in enumerate(contacts) if ids[k] in plan['active_normal_rows']];attempts=[];solution=None
check=lambda q:ns['external'](local,{'p':q.tolist(),'w':(M@q-rhs).tolist()})
class Limit(Exception):pass
class Solved(Exception):pass
for amount in range(len(active)+1):
 for chosen in itertools.combinations(active,amount):
  sticky=[j for j in active if j not in chosen];rows=[contacts[j][0] for j in active]+[int(k) for j in sticky for k in contacts[j][1]];seed=warm[rows];angles=[float(np.arctan2(warm[contacts[j][1][1]],warm[contacts[j][1][0]])) for j in chosen]
  def candidate(theta):
   C=np.zeros((len(warm),len(rows)))
   for ci,k in enumerate(rows):C[k,ci]=1
   for at,j in enumerate(chosen):
    k,ts,mu=contacts[j];C[ts,rows.index(k)]=mu*np.array([np.cos(theta[at]),np.sin(theta[at])])
   B=(M@C)[rows];x=seed+np.linalg.lstsq(B,rhs[rows]-B@seed,rcond=1e-13)[0];q=C@x;w=M@q-rhs
   residual=np.array([np.sin(theta[at])*w[contacts[j][1][0]]-np.cos(theta[at])*w[contacts[j][1][1]] for at,j in enumerate(chosen)])
   return q,residual,float(np.max(abs(w[rows])))
  for offset in ([0.] if not amount else plan['limits']['angle_offsets']):
   for method in (['direct'] if not amount else plan['limits']['methods']):
    box={'count':0,'best':warm.copy(),'score':np.inf,'solution':None,'linear_residual':None}
    def fun(theta):
     if box['count']>=plan['limits']['raw_evaluations_per_start_method']:raise Limit()
     box['count']+=1;q,F,linear=candidate(theta);gate=check(q)
     if gate['projection_m_s']<box['score']:box.update(best=q.copy(),score=gate['projection_m_s'],linear_residual=linear)
     if gate['accepted']:box['solution']=q;raise Solved()
     return F
    try:
     if amount:least_squares(fun,np.array(angles)+offset,method=method,ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=2048)
     else:fun(np.array([]))
    except Limit:pass
    except Solved:pass
    q=box['solution'] if box['solution'] is not None else box['best'];gate=check(q);r={'sliding_normal_rows':[ids[contacts[j][0]] for j in chosen],'offset':offset,'method':method,'raw_evaluations':box['count'],'candidate_p':q.tolist(),'original_gate':gate,'linear_equation_residual':box['linear_residual']};attempts.append(r);save(D/'progress.json',attempts);print(r['sliding_normal_rows'],method,'accepted',gate['accepted'],'residual',gate['projection_m_s'],flush=True)
    if gate['accepted']:solution=q;break
   if solution is not None:break
  if solution is not None:break
 if solution is not None:break
if solution is not None:p[ids]=solution
save(D/'candidate.json',dict(d,p=p.tolist()));native=subprocess.run([str(ROOT/'build/spatial/spatial_coulomb_replay'),str(D/'candidate.json'),'0'],capture_output=True,text=True);(D/'native.stdout.json').write_text(native.stdout);(D/'native.stderr').write_text(native.stderr);gate=ns['external'](d,{'p':p.tolist(),'w':(A@p-b).tolist()});save(D/'summary.json',{'complete':True,'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'accepted':bool(gate['accepted'] and native.returncode==0),'native_exit':native.returncode,'independent':gate,'attempts':attempts})
