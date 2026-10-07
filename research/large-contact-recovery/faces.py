"""Numerical contact-face enumeration with mandatory original full-system gates."""
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from scipy.optimize import least_squares
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
plan=json.loads((H/'face-plan.json').read_text());D=H/'results-faces';D.mkdir(exist_ok=False)
def save(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent-law','exec'),ns)
assert os.environ['OMP_NUM_THREADS']=='1' and os.environ['OPENBLAS_NUM_THREADS']=='1'
class Limit(Exception):pass
class Solved(Exception):pass
for index,(path,sha) in enumerate(plan['inputs'].items()):
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha;data=json.loads((ROOT/path).read_text());A=np.array(data['A']);b=np.array(data['b']);p=np.array(data['p']);dep=np.array(data['dependencies']);hi=np.array(data['hi']);tol=data['tolerance_m_s'];seen=set();attempts=[];declined=False
    for first in range(len(p)):
        if first in seen:continue
        ids=[first];seen.add(first)
        for k in ids:
            for j in range(len(p)):
                if j not in seen and (A[k,j]!=0 or A[j,k]!=0 or dep[k]==j or dep[j]==k):seen.add(j);ids.append(j)
        inverse={k:i for i,k in enumerate(ids)};M=A[np.ix_(ids,ids)];rhs=b[ids];x=p[ids].copy();localdep=np.array([inverse[dep[k]] if dep[k]>=0 else -1 for k in ids]);upper=hi[ids];local={'A':M.tolist(),'b':rhs.tolist(),'dependencies':localdep.tolist(),'hi':upper.tolist(),'tolerance_m_s':tol}
        def check(x):return ns['external'](local,{'p':x.tolist(),'w':(M@x-rhs).tolist()})
        if check(x)['accepted']:continue
        contacts=[]
        for k in np.flatnonzero(localdep<0):
            ts=np.flatnonzero(localdep==k);contacts.append((int(k),ts,np.linalg.eigvalsh(M[np.ix_(ts,ts)])[-1],upper[ts[0]]))
        if len(contacts)>plan['limits']['maximum_contacts_per_component']:declined=True;continue
        masks=list(itertools.chain.from_iterable(itertools.combinations(range(len(contacts)),n) for n in range(len(contacts)+1)))[:plan['limits']['maximum_face_masks_per_failed_component']];recovered=False
        for mask in masks:
            active=[c for i,c in enumerate(contacts) if i not in mask];rows=np.array(sorted(k for c in active for k in np.r_[c[0],c[1]]),dtype=int)
            if not len(rows):continue
            seed=np.linalg.lstsq(M[np.ix_(rows,rows)],rhs[rows],rcond=plan['limits']['singular_cutoff_for_linear_seed'])[0];position={k:i for i,k in enumerate(rows)}
            for c in active:seed[position[c[0]]]=max(1e-14,seed[position[c[0]]])
            lower=np.full(len(rows),-np.inf)
            for c in active:lower[position[c[0]]]=0
            for method in plan['limits']['methods']:
                box={'count':0,'best':x.copy(),'score':np.inf,'solution':None}
                def fun(z):
                    if box['count']>=plan['limits']['raw_function_evaluations_per_method_mask']:raise Limit()
                    box['count']+=1;q=np.zeros(len(x));q[rows]=z;w=M@q-rhs;F=np.zeros(len(x))
                    for k,ts,eig,mu in active:
                        F[k]=w[k];v=q[ts]-w[ts]/eig;cap=mu*max(0,q[k]);F[ts]=(q[ts]-v*min(1,cap/max(np.linalg.norm(v),1e-300)))*eig
                    score=float(np.max(np.abs(F[rows])))
                    if score<box['score']:box['score']=score;box['best']=q.copy()
                    if score<=tol and check(q)['accepted']:box['solution']=q;raise Solved()
                    return F[rows]
                try:least_squares(fun,seed,method=method,x_scale='jac',bounds=(lower,np.full(len(rows),np.inf)) if method=='trf' else (-np.inf,np.inf),ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=2048)
                except Limit:pass
                except Solved:pass
                candidate=box['solution'] if box['solution'] is not None else box['best'];gate=check(candidate);record={'original_component_rows':ids,'inactive_contacts':[ids[contacts[i][0]] for i in mask],'method':method,'raw_evaluations':box['count'],'candidate_p':candidate.tolist(),'original_component_gate':gate};attempts.append(record);save(D/'progress.json',attempts)
                print(index,'mask',[ids[contacts[i][0]] for i in mask],method,'accepted',gate['accepted'],'residual',gate['projection_m_s'],flush=True)
                if gate['accepted']:p[ids]=candidate;recovered=True;break
            if recovered:break
        declined |= not recovered
    gate=ns['external'](data,{'p':p.tolist(),'w':(A@p-b).tolist()});candidate=dict(data,p=p.tolist());capture=D/'candidate.json';save(capture,candidate);native=subprocess.run([str(ROOT/'build/spatial/spatial_coulomb_replay'),str(capture),'0'],capture_output=True,text=True);(D/'native.stdout.json').write_text(native.stdout);(D/'native.stderr').write_text(native.stderr)
    save(D/'summary.json',{'complete':True,'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'input':path,'declined':declined,'accepted':bool(gate['accepted'] and native.returncode==0),'independent':gate,'native_exit':native.returncode,'attempts':attempts})
