from pathlib import Path
import json,hashlib,os,time,subprocess
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
import numpy as np
from scipy.optimize import least_squares,root
p=Path(__file__).resolve().parent;plan=json.loads((p/'plan.json').read_text());data=json.loads(Path(plan['capture']).read_text());A=np.array(data['A']);b=np.array(data['b']);initial=np.array(data['p']);dep=np.array(data['dependencies']);hi=np.array(data['hi']);tol=data['tolerance_m_s'];n=len(b)
cs=[]
for k in np.flatnonzero(dep<0):
 t,s=np.flatnonzero(dep==k);block=A[np.ix_([t,s],[t,s])];cs.append((int(k),int(t),int(s),hi[t],np.linalg.eigvalsh(block)[-1]))
def fj(x):
 w=A@x-b;F=np.zeros(n);J=np.zeros((n,n))
 for k,t,s,mu,eig in cs:
  zn=x[k]-w[k]/A[k,k];F[k]=(x[k]-max(0.,zn))*A[k,k]
  J[k]=A[k] if zn>0 else np.eye(n)[k]*A[k,k]
  rows=[t,s];z=x[rows]-w[rows]/eig;length=np.linalg.norm(z);cap=mu*max(0.,x[k])
  if length<=cap and cap>0:F[rows]=w[rows];J[rows]=A[rows]
  else:
   direction=z/length if length>0 else np.zeros(2)
   F[rows]=(x[rows]-cap*direction)*eig
   D=cap/length*(np.eye(2)-np.outer(direction,direction)) if length>0 else np.zeros((2,2))
   J[rows]=eig*(np.eye(n)[rows]-D@(np.eye(n)[rows]-A[rows]/eig))
   if x[k]>0:J[rows,k]-=eig*mu*direction
 return F,J
from scipy.sparse.csgraph import connected_components
nc,labels=connected_components(A!=0,directed=False)
evals=np.linalg.eigvalsh(A)
structure={'matrix_eigenvalues':evals.tolist(),'rank_relative_1e12':int(np.sum(evals>evals[-1]*1e-12)),'components':[np.flatnonzero(labels==i).tolist() for i in range(nc)],'initial_projection_residuals':fj(initial)[0].tolist(),'diagonal':np.diag(A).tolist()}
(p/'structure.json').write_text(json.dumps(structure,indent=2)+'\n')
print('eigs',evals[0],evals[-1],'rank',structure['rank_relative_1e12'],'components',list(map(len,structure['components'])),flush=True)
seeds={'captured':initial,'zero':np.zeros(n),'pinv':np.linalg.pinv(A,rcond=1e-12)@b}
for x in seeds.values():
 for k,t,s,mu,eig in cs:
  x[k]=max(0.,x[k]);cap=mu*x[k];r=np.linalg.norm(x[[t,s]])
  if r>cap:x[[t,s]]*=cap/r
small=initial.copy()
for k,t,s,mu,eig in cs:
 if small[k]<1e-3:small[[k,t,s]]=0
seeds['small_contacts_open']=small
summary=[]
for seed_name,x0 in seeds.items():
 for method in ['lm','trf']:
  start=time.perf_counter();r=least_squares(lambda x:fj(x)[0],x0.copy(),jac=lambda x:fj(x)[1],method=method,ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=3000,x_scale='jac');res=fj(r.x)[0]
  outcome={'seed':seed_name,'method':method,'status':int(r.status),'message':r.message,'nfev':int(r.nfev),'njev':None if r.njev is None else int(r.njev),'projection_max':float(max(abs(res))),'seconds_descriptive':time.perf_counter()-start,'p':r.x.tolist(),'w':(A@r.x-b).tolist(),'F':res.tolist(),'cost':float(r.cost),'optimality':float(r.optimality)}
  name=seed_name+'-'+method;(p/(name+'.json')).write_text(json.dumps(outcome,indent=2)+'\n');summary.append({k:v for k,v in outcome.items() if k not in ['p','w','F']});print(name,outcome['projection_max'],outcome['nfev'],flush=True)
  if outcome['projection_max']<=tol:
   native_input=dict(data,p=r.x.tolist());ip=p/(name+'-native-input.json');ip.write_text(json.dumps(native_input)+'\n');exe=Path('build/spatial/spatial_coulomb_replay').resolve();nr=subprocess.run([str(exe),str(ip),'0'],capture_output=True,text=True);(p/(name+'-native.stdout.json')).write_text(nr.stdout);(p/(name+'-native.stderr')).write_text(nr.stderr);(p/(name+'-native-exit.json')).write_text(json.dumps({'exit':nr.returncode})+'\n');print('native',nr.returncode,flush=True)
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(p/'guards-after.json').write_text(json.dumps({'unchanged':all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in plan['guards_before'].items()),'guards':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in plan['guards_before']}},indent=2)+'\n')
