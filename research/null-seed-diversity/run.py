"""Bounded unchanged-law pressure-gauge seeds and independent/native acceptance."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.optimize import least_squares
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
plan=json.loads((H/'plan.json').read_text());D=H/'results';D.mkdir(exist_ok=False)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
source=ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py'
fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external')
ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'original-independent-law','exec'),ns)
assert os.environ['OPENBLAS_NUM_THREADS']=='1' and os.environ['OMP_NUM_THREADS']=='1'
class Limit(Exception):pass
class Solved(Exception):pass
all_records=[]
for index,(path,sha) in enumerate(plan['inputs'].items()):
    assert digest(ROOT/path)==sha;data=json.loads((ROOT/path).read_text());A=np.array(data['A']);b=np.array(data['b']);x=np.array(data['p']);dep=np.array(data['dependencies']);hi=np.array(data['hi']);tol=data['tolerance_m_s'];seen=set();components=[];attempts=[]
    for first in range(len(b)):
        if first in seen:continue
        ids=[first];seen.add(first)
        for k in ids:
            for j in range(len(b)):
                if j not in seen and (A[k,j]!=0 or A[j,k]!=0 or dep[k]==j or dep[j]==k):seen.add(j);ids.append(j)
        components.append(ids)
    failed=False
    for ci,ids in enumerate(components):
        inv={k:i for i,k in enumerate(ids)};M=A[np.ix_(ids,ids)];rhs=b[ids];p=x[ids].copy();local_dep=np.array([inv[dep[k]] if dep[k]>=0 else -1 for k in ids]);upper=hi[ids]
        local={'A':M.tolist(),'b':rhs.tolist(),'dependencies':local_dep.tolist(),'hi':upper.tolist(),'tolerance_m_s':tol};contacts=[]
        for k in np.flatnonzero(local_dep<0):
            ts=np.flatnonzero(local_dep==k);eig=np.linalg.eigvalsh(M[np.ix_(ts,ts)])[-1];contacts.append((int(k),ts,eig,upper[ts[0]]))
        def clean(q):
            q=q.copy()
            for k,ts,eig,mu in contacts:
                if -tol/M[k,k]<=q[k]<0:q[k]=0
            return q
        def check(q):return ns['external'](local,{'p':q.tolist(),'w':(M@q-rhs).tolist()})
        def equations(q):
            w=M@q-rhs;F=np.zeros(len(q))
            for k,ts,eig,mu in contacts:
                F[k]=(q[k]-max(0,q[k]-w[k]/M[k,k]))*M[k,k]
                z=q[ts]-w[ts]/eig;cap=mu*max(0,q[k]);length=np.linalg.norm(z)
                F[ts]=(q[ts]-z*min(1,cap/max(length,1e-300)))*eig
            return F
        if check(p)['accepted']:continue
        U,S,V=np.linalg.svd(M);Q=V[S<=S[0]*plan['limits']['mobility_null_relative_singular_cutoff']].T
        F=equations(p);ordered=sorted(contacts,key=lambda c:max(abs(F[c[0]]),np.linalg.norm(F[c[1]])),reverse=True)
        seeds=[('warm',p)]
        for k,ts,eig,mu in ordered[:plan['limits']['maximum_targeted_contact_seeds_per_failed_component']]:
            if Q.shape[1]==0:break
            rows=np.r_[k,ts];coeff=np.linalg.lstsq(Q[rows],-p[rows],rcond=1e-12)[0];trial=p+Q@coeff;trial[rows]=0
            for j,tt,ee,mm in contacts:
                trial[j]=max(0,trial[j]);length=np.linalg.norm(trial[tt]);trial[tt]*=min(1,mm*trial[j]/max(length,1e-300))
            seeds.append((f'drop_{ids[k]}',trial))
        # Deterministic search-only diversity in the numerical mobility nullspace.
        rng=np.random.default_rng(plan['limits']['random_seed']+index*100+ci)
        for sample in range(plan['limits']['random_null_seeds']):
            if Q.shape[1]==0:break
            direction=Q@rng.normal(size=Q.shape[1]);direction/=max(np.linalg.norm(direction),1e-300)
            magnitude=max(1e-5,np.linalg.norm(p))*10.**(-2+sample%6)
            trial=p+direction*magnitude
            for j,tt,ee,mm in contacts:
                trial[j]=max(0,trial[j]);length=np.linalg.norm(trial[tt]);trial[tt]*=min(1,mm*trial[j]/max(length,1e-300))
            seeds.append((f'random_null_{sample}',trial))
        recovered=False
        for label,seed in seeds:
            for method in plan['limits']['methods']:
                count=0;best=clean(seed);best_res=float(np.max(np.abs(equations(best))));started=time.perf_counter();status='limit_or_decline'
                # Mutable counters retain every actual finite-difference evaluation.
                box={'count':0,'best':best,'score':best_res,'solution':None}
                def fun(q):
                    if box['count']>=plan['limits']['raw_function_evaluations_per_method_seed']:raise Limit()
                    box['count']+=1;F=equations(q);score=float(np.max(np.abs(F)))
                    if score<box['score']:box['score']=score;box['best']=q.copy()
                    if score<=tol:
                        candidate=clean(q)
                        if check(candidate)['accepted']:box['solution']=candidate;raise Solved()
                    return F
                try:least_squares(fun,seed,method=method,x_scale='jac',ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=2048)
                except Solved:status='accepted'
                except Limit:status='raw_evaluation_cap'
                candidate=clean(box['solution'] if box['solution'] is not None else box['best']);gate=check(candidate)
                attempt={'component':ci,'original_rows':ids,'seed':label,'seed_p':seed.tolist(),'seed_response_change_max':float(np.max(np.abs(M@(seed-p)))),'method':method,'raw_evaluations':box['count'],'status':status,'elapsed_s_descriptive':time.perf_counter()-started,'candidate_p':candidate.tolist(),'independent':gate,'null_dimension':Q.shape[1]};attempts.append(attempt);save(D/f'{index}-progress.json',attempts)
                print(index,ci,label,method,'accepted',gate['accepted'],'residual',gate['projection_m_s'],flush=True)
                if gate['accepted']:x[ids]=candidate;recovered=True;break
            if recovered:break
        failed |= not recovered
    gate=ns['external'](data,{'p':x.tolist(),'w':(A@x-b).tolist()});modified=dict(data,p=x.tolist());capture=D/f'{index}-candidate.json';save(capture,modified)
    native=subprocess.run([str(ROOT/'build/spatial/spatial_coulomb_replay'),str(capture),'0'],capture_output=True,text=True);(D/f'{index}-native.stdout.json').write_text(native.stdout);(D/f'{index}-native.stderr').write_text(native.stderr)
    accepted=gate['accepted'] and native.returncode==0;record={'input':path,'sha256':sha,'component_search_declined':failed,'independent':gate,'native_exit':native.returncode,'accepted':bool(accepted),'attempts':attempts};all_records.append(record);save(D/'progress.json',all_records)
    print('FULL',index,'accepted',accepted,flush=True)
save(D/'summary.json',{'complete':True,'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'records':all_records,'scope':'Instantaneous-system proof only; production/world/trajectory qualification remains separate.'})
