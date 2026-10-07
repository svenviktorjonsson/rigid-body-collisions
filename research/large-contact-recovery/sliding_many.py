"""Numerical exact circular-cone face parameters; original physical acceptance."""
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from scipy.optimize import least_squares
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'sliding-many-plan.json').read_text());D=H/'results-sliding-many';D.mkdir(exist_ok=False)
def save(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent-law','exec'),ns)
assert os.environ['OMP_NUM_THREADS']=='1' and os.environ['OPENBLAS_NUM_THREADS']=='1'
class Limit(Exception):pass
class Solved(Exception):pass
for index,(path,sha) in enumerate(plan['inputs'].items()):
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha;d=json.loads((ROOT/path).read_text());A=np.array(d['A']);b=np.array(d['b']);p=np.array(d['p']);dep=np.array(d['dependencies']);hi=np.array(d['hi']);tol=d['tolerance_m_s'];seen=set();attempts=[];declined=False
    for first in range(len(p)):
        if first in seen:continue
        ids=[first];seen.add(first)
        for k in ids:
            for j in range(len(p)):
                if j not in seen and (A[k,j]!=0 or A[j,k]!=0 or dep[k]==j or dep[j]==k):seen.add(j);ids.append(j)
        inverse={k:i for i,k in enumerate(ids)};M=A[np.ix_(ids,ids)];rhs=b[ids];warm=p[ids].copy();localdep=np.array([inverse[dep[k]] if dep[k]>=0 else -1 for k in ids]);upper=hi[ids];local={'A':M.tolist(),'b':rhs.tolist(),'dependencies':localdep.tolist(),'hi':upper.tolist(),'tolerance_m_s':tol}
        def check(q):return ns['external'](local,{'p':q.tolist(),'w':(M@q-rhs).tolist()})
        if check(warm)['accepted']:continue
        contacts=[]
        for k in np.flatnonzero(localdep<0):
            ts=np.flatnonzero(localdep==k);contacts.append((int(k),ts,np.linalg.eigvalsh(M[np.ix_(ts,ts)])[-1],upper[ts[0]]))
        starts=0;recovered=False;warmw=M@warm-rhs
        for amount in range(3,plan['limits']['max_selected_sliding_contacts']+1):
            for chosen in itertools.combinations(range(len(contacts)),amount):
                if starts>=plan['limits']['max_starts']:break
                selected={contacts[j][0] for j in chosen};removed={int(k) for j in chosen for k in contacts[j][1]};keep=np.array([j for j in range(len(warm)) if j not in removed]);position={k:i for i,k in enumerate(keep)}
                offsets=[[0.0]*amount,[float(np.pi)]*amount]
                for shift in offsets:
                    if starts>=plan['limits']['max_starts']:break
                    starts+=1;angles=[float(np.arctan2(-warmw[contacts[j][1][1]],-warmw[contacts[j][1][0]])+s) for j,s in zip(chosen,shift)];seed=np.r_[warm[keep],angles];lower=np.full(len(seed),-np.inf)
                    for k,ts,eig,mu in contacts:lower[position[k]]=0;seed[position[k]]=max(seed[position[k]],1e-14)
                    def unpack(z):
                        q=np.zeros(len(warm));q[keep]=z[:len(keep)]
                        for at,j in enumerate(chosen):
                            k,ts,eig,mu=contacts[j];angle=z[len(keep)+at];q[ts]=mu*max(0,q[k])*np.array([np.cos(angle),np.sin(angle)])
                        return q
                    for method in plan['limits']['methods']:
                        box={'count':0,'best':warm.copy(),'score':np.inf,'solution':None}
                        def fun(z):
                            if box['count']>=plan['limits']['raw_function_evaluations_per_method_start']:raise Limit()
                            box['count']+=1;q=unpack(z);w=M@q-rhs;F=[]
                            for k,ts,eig,mu in contacts:
                                F.append((q[k]-max(0,q[k]-w[k]/M[k,k]))*M[k,k])
                                if k not in selected:
                                    v=q[ts]-w[ts]/eig;cap=mu*max(0,q[k]);F.extend((q[ts]-v*min(1,cap/max(np.linalg.norm(v),1e-300)))*eig)
                            for at,j in enumerate(chosen):
                                k,ts,eig,mu=contacts[j];angle=z[len(keep)+at];F.append(np.sin(angle)*w[ts[0]]-np.cos(angle)*w[ts[1]] if q[k]>0 else 0.)
                            gate=check(q);score=gate['projection_m_s']
                            if score<box['score']:box['score']=score;box['best']=q.copy()
                            if gate['accepted']:box['solution']=q;raise Solved()
                            return np.array(F)
                        try:least_squares(fun,seed,method=method,x_scale='jac',bounds=(lower,np.full(len(seed),np.inf)) if method=='trf' else (-np.inf,np.inf),ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=2048)
                        except Limit:pass
                        except Solved:pass
                        q=box['solution'] if box['solution'] is not None else box['best'];gate=check(q);record={'component_rows':ids,'sliding_contacts':[ids[contacts[j][0]] for j in chosen],'angle_offsets':shift,'method':method,'raw_evaluations':box['count'],'candidate_p':q.tolist(),'original_gate':gate};attempts.append(record);save(D/'progress.json',attempts);print('sliding',record['sliding_contacts'],shift,method,'accepted',gate['accepted'],'residual',gate['projection_m_s'],flush=True)
                        if gate['accepted']:p[ids]=q;recovered=True;break
                    if recovered:break
                if recovered:break
            if recovered:break
        declined |= not recovered
    gate=ns['external'](d,{'p':p.tolist(),'w':(A@p-b).tolist()});capture=D/'candidate.json';save(capture,dict(d,p=p.tolist()));native=subprocess.run([str(ROOT/'build/spatial/spatial_coulomb_replay'),str(capture),'0'],capture_output=True,text=True);(D/'native.stdout.json').write_text(native.stdout);(D/'native.stderr').write_text(native.stderr);save(D/'summary.json',{'complete':True,'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'accepted':bool(gate['accepted'] and native.returncode==0),'declined':declined,'native_exit':native.returncode,'independent':gate,'attempts':attempts})
