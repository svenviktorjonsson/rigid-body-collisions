"""Exact finite normal-support enumeration of sticking-only candidate faces."""
import ast,hashlib,itertools,json,subprocess
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'results';D.mkdir(exist_ok=False);plan=json.loads((H/'plan.json').read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(ROOT/plan['input'])==plan['input_sha256'];data=json.loads((ROOT/plan['input']).read_text());A=np.array(data['A']);b=np.array(data['b']);p=np.array(data['p']);dep=np.array(data['dependencies']);hi=np.array(data['hi']);tol=data['tolerance_m_s'];ids=plan['component_rows'];M=A[np.ix_(ids,ids)];rhs=b[ids];inv={k:j for j,k in enumerate(ids)};d=np.array([inv[k] if k>=0 else -1 for k in dep[ids]]);upper=hi[ids];local=dict(A=M.tolist(),b=rhs.tolist(),hi=upper.tolist(),dependencies=d.tolist(),tolerance_m_s=tol)
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent','exec'),ns);normals=np.flatnonzero(d<0);assert len(normals)<=plan['maximum_normals'];attempts=[];accepted=False
for bits in itertools.product([False,True],repeat=len(normals)):
 active=[k for k,on in zip(normals,bits) if on];rows=[j for j in range(len(rhs)) if j in active or d[j] in active];q=np.zeros(len(rhs));rank=0
 if rows:
  value,residual,rank,s=np.linalg.lstsq(M[np.ix_(rows,rows)],rhs[rows],rcond=1e-12);q[rows]=value
 for k in normals:
  if -tol/M[k,k]<=q[k]<0:q[k]=0
 gate=ns['external'](local,dict(p=q.tolist(),w=(M@q-rhs).tolist()));attempts.append({'active_original_normals':[ids[k] for k in active],'linear_rank':int(rank),'candidate_p':q.tolist(),'independent':gate})
 if gate['accepted']:p[ids]=q;accepted=True;break
save(D/'attempts.json',attempts);capture=D/'candidate.json';save(capture,dict(data,p=p.tolist()));out=subprocess.run([str(ROOT/'build/spatial/spatial_coulomb_replay'),str(capture),'0'],capture_output=True,text=True);(D/'native.stdout.json').write_text(out.stdout);(D/'native.stderr').write_text(out.stderr);gate=ns['external'](data,dict(p=p.tolist(),w=(A@p-b).tolist()));save(D/'summary.json',{'complete':True,'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'attempted_supports':len(attempts),'total_sticking_supports':2**len(normals),'accepted':bool(accepted and gate['accepted'] and out.returncode==0),'independent':gate,'native_exit':out.returncode,'scope':'Sticking-only linear faces; failure is NOT proof no sliding original-law root exists. No body, world, reference or performance acceptance.'});print('Sticking supports',len(attempts),'accepted',accepted and gate['accepted'] and out.returncode==0)
