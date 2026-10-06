"""Bounded mixed sliding/sticking faces, conservative inner cone LP seeds."""
import ast,hashlib,json,os,subprocess,time,itertools
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'plan.json').read_text());D=H/'results';D.mkdir(exist_ok=False)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'original-gate','exec'),ns);external=ns['external'];exe=ROOT/'build/spatial/spatial_coulomb_replay';paths=[Path(__file__),H/'plan.json',exe,*[p for p in (ROOT/'spatial_backend').glob('*') if p.is_file()]];guards={str(p):sha(p) for p in paths};save(D/'provenance.json',{'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'guards':guards,'scope':'Read-only numerical comparator; candidates never applied to a world.'});records=[]
for index,(path,digest) in enumerate(plan['inputs'].items()):
 assert sha(ROOT/path)==digest;data=json.loads((ROOT/path).read_text());A=np.array(data['A']);b=np.array(data['b']);p=np.array(data['p']);dep=np.array(data['dependencies']);hi=np.array(data['hi']);candidate=p.copy();todo=set(range(len(b)));lp_calls=0;found=True;parts=[];start=time.perf_counter()
 while todo:
  ids=[todo.pop()]
  for i in ids:
   new={j for j in todo if A[i,j]!=0 or A[j,i]!=0 or dep[j]==i or dep[i]==j};ids+=sorted(new);todo-=new
  m=len(ids);M=A[np.ix_(ids,ids)];rhs=b[ids];q=candidate[ids].copy();inv={j:i for i,j in enumerate(ids)};d=np.array([inv[dep[j]] if dep[j]>=0 else -1 for j in ids]);normals=np.flatnonzero(d<0);w=M@q-rhs
  local={k:np.array(data[k])[ids].tolist() for k in ['lo','hi']};local.update(A=M.tolist(),b=rhs.tolist(),dependencies=d.tolist(),tolerance_m_s=data['tolerance_m_s'])
  if external(local,{'p':q.tolist(),'w':w.tolist()})['accepted']:continue
  active=[k for k in normals if q[k]>data['tolerance_m_s']/M[k,k] or w[k]<-data['tolerance_m_s']];rows=sorted([i for i in range(m) if i in active or d[i] in active]);r=len(rows);where={j:i for i,j in enumerate(rows)};weak=[k for k in active if np.linalg.norm(w[np.flatnonzero(d==k)])<=plan['near_sticking_speed_m_s']];strong=[k for k in active if k not in weak]
  part={'component_rows':m,'search_rows':r,'weak_contacts':len(weak),'strong_sliding_contacts':len(strong)};parts.append(part)
  if not rows or r>192 or len(weak)>8:part['decline']='declared search/weak-face cap';found=False;break
  directions={k:w[np.flatnonzero(d==k)]/np.linalg.norm(w[np.flatnonzero(d==k)]) for k in strong};ok=False
  for bits in itertools.product([True,False],repeat=len(weak)):
   dropped={k for k,keep in zip(weak,bits) if not keep};stick=set(weak)-dropped;mode_dirs={k:v.copy() for k,v in directions.items()}
   for iteration in range(128):
    if lp_calls>=4096:break
    eq=[];erhs=[];ub=[];urhs=[];bounds=[(None,None)]*r;cost=np.zeros(r)
    for k in normals:
     if k in active and k not in dropped:eq.append(M[k,rows]);erhs.append(rhs[k])
     else:ub.append(-M[k,rows]);urhs.append(-rhs[k])
    for k in active:
     nk=where[k];t=list(np.flatnonzero(d==k));mu=hi[ids[t[0]]];cost[nk]=1;bounds[nk]=(0,None)
     if k in dropped:
      for j in [k,*t]:bounds[where[j]]=(0,0)
     elif k in stick:
      for j in t:eq.append(M[j,rows]);erhs.append(rhs[j])
      for angle in np.arange(32)*2*np.pi/32:
       line=np.zeros(r);line[where[t[0]]]=np.cos(angle);line[where[t[1]]]=np.sin(angle);line[nk]=-mu*np.cos(np.pi/32);ub.append(line);urhs.append(0.)
     else:
      for axis,j in enumerate(t):
       line=np.zeros(r);line[where[j]]=1;line[nk]=mu*mode_dirs[k][axis];eq.append(line);erhs.append(0.)
    lp_calls+=1;fit=linprog(cost,A_ub=np.array(ub),b_ub=np.array(urhs),A_eq=np.array(eq),b_eq=np.array(erhs),bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'presolve':True})
    if not fit.success:break
    trial=np.zeros(m);trial[rows]=fit.x
    for k in normals:
     trial[k]=max(0,trial[k]);t=np.flatnonzero(d==k);cap=hi[ids[t[0]]]*trial[k];length=np.linalg.norm(trial[t]);
     if length>cap:trial[t]*=cap/length
    response=M@trial-rhs;gate=external(local,{'p':trial.tolist(),'w':response.tolist()})
    if gate['accepted']:
     candidate[ids]=trial;ok=True;part.update(accepted=True,lp_calls=lp_calls,dropped_contacts=sorted(dropped),original_gate=gate);break
    invalid=False
    for k in strong:
     v=response[np.flatnonzero(d==k)];length=np.linalg.norm(v)
     if length<=1e-12:invalid=True;break
     direction=.5*mode_dirs[k]+.5*v/length;mode_dirs[k]=direction/np.linalg.norm(direction)
    if invalid:break
   if ok or lp_calls>=4096:break
  if not ok:part.update(accepted=False,lp_calls=lp_calls);found=False;break
 gate=external(data,{'p':candidate.tolist(),'w':(A@candidate-b).tolist()});verified=False
 if found and gate['accepted']:
  dump=dict(data,p=candidate.tolist());save(D/f'{index}.candidate.json',dump);r=subprocess.run([str(exe),str(D/f'{index}.candidate.json'),'0'],capture_output=True,text=True);(D/f'{index}.native.stdout.json').write_text(r.stdout);(D/f'{index}.native.stderr').write_text(r.stderr);verified=r.returncode==0 and json.loads(r.stdout)['accepted'];assert verified
 record={'input':path,'accepted':bool(verified),'lp_calls':lp_calls,'parts':parts,'original_full_gate':gate,'elapsed_s_descriptive':time.perf_counter()-start};records.append(record);save(D/'progress.json',records);print(path,'accepted',verified,'LPs',lp_calls,'parts',parts,flush=True)
assert all(sha(p)==v for p,v in guards.items());save(D/'summary.json',{'complete':True,'guards_unchanged':True,'records':records,'scope':'Mixed-face LP search only; full unchanged original gate and native C0 mandatory. No world/performance qualification.'})
