from pathlib import Path
import json,os,time,hashlib,subprocess
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
import numpy as np
from scipy.optimize import least_squares
p=Path(__file__).resolve().parent;plan=json.loads((p/'plan.json').read_text());d=json.loads(Path(plan['capture']).read_text());A=np.array(d['A']);b=np.array(d['b']);x=np.array(d['p']);dep=np.array(d['dependencies']);hi=np.array(d['hi']);w=A@x-b
spec={'__file__':str(p/'search.py')};exec((p/'search.py').read_text().split('from scipy.sparse.csgraph')[0],spec);fj=spec['fj'];seeds=[]
for k in np.flatnonzero(dep<0):
 if x[k]<=0:continue
 for t in np.flatnonzero(dep>=0):
  sign=1 if np.array_equal(A[k],A[t]) and b[k]==b[t] else (-1 if np.array_equal(A[k],-A[t]) and b[k]==-b[t] else 0)
  if not sign:continue
  c=dep[t];ts=np.flatnonzero(dep==c);s=int(ts[0] if ts[1]==t else ts[1]);candidate=x.copy();candidate[t]=0;candidate[s]=-np.copysign(hi[s]*max(0,x[c]),w[s]);seeds.append({'normal':int(k),'tangent':int(t),'sign':sign,'s':s,'seed':candidate.tolist()})
(p/'structural-seeds.json').write_text(json.dumps(seeds,indent=2)+'\n');print('seeds',[(s['normal'],s['tangent']) for s in seeds],flush=True)
if len(seeds)!=1:raise RuntimeError('Unexpected number of equalities; preserve plan cap')
x0=np.array(seeds[0]['seed']);ip=p/'structural-native-input.json';ip.write_text(json.dumps(dict(d,p=x0.tolist()))+'\n');exe=Path('build/spatial/spatial_coulomb_replay').resolve()
for budget in [256,4096]:
 r=subprocess.run([str(exe),str(ip),str(budget)],capture_output=True,text=True);(p/f'structural-native-{budget}.stdout.json').write_text(r.stdout);(p/f'structural-native-{budget}.stderr').write_text(r.stderr);(p/f'structural-native-{budget}.exit.json').write_text(json.dumps({'returncode':r.returncode})+'\n');out=json.loads(r.stdout);print('native',budget,r.returncode,out['stats']['residual_m_s'],out['stats']['iteration_sweeps_total'],flush=True)
for method in ['lm','trf']:
 count=0;first=None
 def fun(z):
  global count,first
  F,J=fj(z);count+=1
  if first is None and max(abs(F))<=d['tolerance_m_s']:first=count
  return F
 r=least_squares(fun,x0,jac=lambda z:fj(z)[1],method=method,ftol=1e-14,xtol=1e-14,gtol=1e-14,x_scale='jac',max_nfev=3000)
 out={'p':r.x.tolist(),'status':int(r.status),'nfev':int(r.nfev),'njev':int(r.njev),'projection_max':float(max(abs(fj(r.x)[0]))),'first_projection_gate_function_call':first};(p/f'structural-{method}.json').write_text(json.dumps(out,indent=2)+'\n');print(method,{k:v for k,v in out.items() if k!='p'},flush=True)
 if out['projection_max']<=d['tolerance_m_s']:
  ci=p/f'structural-{method}-native-input.json';ci.write_text(json.dumps(dict(d,p=r.x.tolist()))+'\n');nr=subprocess.run([str(exe),str(ci),'0'],capture_output=True,text=True);(p/f'structural-{method}-native.stdout.json').write_text(nr.stdout);(p/f'structural-{method}-native.stderr').write_text(nr.stderr);(p/f'structural-{method}-native-exit.json').write_text(json.dumps({'returncode':nr.returncode})+'\n')
(p/'structural-guards-after.json').write_text(json.dumps({'unchanged':all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in plan['guards_before'].items())},indent=2)+'\n')
