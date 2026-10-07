"""Paired same-law local experiment; run from repository root."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from scipy.optimize import lsq_linear

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from contact_history import ContactHistory

spec=importlib.util.spec_from_file_location('frozen_history',Path(__file__).with_name('frozen_contact_history.py'))
frozen=importlib.util.module_from_spec(spec);sys.modules[spec.name]=frozen;spec.loader.exec_module(frozen)


def verification():
    rng=np.random.default_rng(20261007);maximum=0.;optim_error=0.;count=0;sliding=0;optimizer_cases=0
    for n in (1,2,3):
        for k in range(500):
            R=rng.normal(size=(n,n));G=R@R.T
            if k%11==0:G=np.ones((n,n)) # PSD with a null space
            K=10**rng.uniform(0,4,n);h=10**rng.uniform(-3,-1)
            u=rng.normal(size=n)*2;eta=rng.normal(size=n)*.02
            cd=10**rng.uniform(-3,0,n);cs=cd*1.5
            if k%5==0:cd[k%n]=cs[k%n]=0
            old=frozen.ContactHistory(G,K,h);new=ContactHistory(G,K,h)
            reference=old.step(u,eta,cs,cd,active=k%17!=0)
            candidate=new.step(u,eta,cs,cd,active=k%17!=0,active_set=tuple(int(s) for s in rng.choice([-1,0,1,2],n)))
            for field in dataclasses.fields(reference):
                a=np.asarray(getattr(reference,field.name),dtype=float);b=np.asarray(getattr(candidate,field.name),dtype=float)
                error=float(np.max(np.abs(a-b)/np.maximum(1.,np.abs(a))))
                maximum=max(maximum,error)
                np.testing.assert_allclose(a,b,rtol=2e-10,atol=1e-11)
            count+=1;sliding+=candidate.sliding
            if candidate.sliding and k%10==0:
                optimizer_cases+=1
                A=new.A;b=h*u+2*eta
                # A=L L^T, so min 1/2 ||L^T j + L^-1 b||^2 is the
                # same convex quadratic up to a constant. BVLS is an independent
                # library optimizer; eliminate singleton zero-capacity variables.
                L=np.linalg.cholesky(A);free=np.flatnonzero(cd>0);prediction=np.zeros(n)
                if len(free):
                    result=lsq_linear(L.T[:,free],-np.linalg.solve(L,b),
                        bounds=(-cd[free],cd[free]),method='bvls',tol=1e-13,max_iter=500)
                    if not result.success:raise RuntimeError(result.message)
                    prediction[free]=result.x
                error=float(np.max(np.abs(prediction-candidate.impulse)))
                optim_error=max(optim_error,error)
                # Compare objective and impulse: independent optimizer has a
                # looser numerical tolerance than exact active-set enumeration.
                np.testing.assert_allclose(prediction,candidate.impulse,rtol=1e-8,atol=1e-10)
    return dict(cases=count,sliding_cases=int(sliding),max_scaled_frozen_difference=maximum,
                independent_optimizer_cases=optimizer_cases,independent_optimizer_max_impulse_error=optim_error)


def benchmark(sizes,scenarios):
    rows=[]
    for n in (1,2,3):
        for scenario in scenarios:
            if scenario=='coupled_face' and n==1:continue
            G=np.eye(n)*1.5+np.ones((n,n))*.2;K=np.linspace(200.,1000.,n);h=.02
            if scenario=='coupled_face':
                G=np.eye(n);G[:2,:2]=[[1.,.8],[.8,1.]];K[:]=4.;h=1.
            start=time.perf_counter();old=frozen.ContactHistory(G,K,h);old_prep=time.perf_counter()-start
            start=time.perf_counter();new=ContactHistory(G,K,h);new_prep=time.perf_counter()-start
            cd=np.full(n,.002);cs=cd*1.5
            if scenario=='stick':cd[:]=cs[:]=100
            u=np.arange(1,n+1,dtype=float)*2;eta=np.zeros(n)
            if scenario=='coupled_face':cd[:]=.1;cs[:]=.15;u[:]=.05;u[:2]=[1.,.1]
            if scenario=='coherent_upper':u=-u
            if scenario=='coherent_mixed':u[::2]*=-1
            hint=new.step(u,eta,cs,cd).active_set
            for count in sizes:
                inputs=tuple((u if scenario!='reversing_slide' or k%2==0 else -u,eta,cs,cd) for k in range(count))
                def run(law,warm=False):
                    state=hint;checksum=0.
                    start=time.perf_counter()
                    for args in inputs:
                        result=law.step(*args,active_set=state) if warm else law.step(*args)
                        checksum+=float(result.impulse[0])+result.stored_energy+result.plastic_loss
                        if warm:state=result.active_set
                    return (time.perf_counter()-start)*1000,checksum
                run(old);run(new);run(new,True)
                samples={'frozen':[],'candidate_cold':[],'candidate_hint':[]};checks={}
                entries=[('frozen',old,False),('candidate_cold',new,False),('candidate_hint',new,True)]
                for repeat in range(7):
                    for name,law,warm in (entries if repeat%2==0 else entries[::-1]):
                        duration,checksum=run(law,warm);samples[name].append(duration);checks[name]=checksum
                np.testing.assert_allclose(list(checks.values()),checks['frozen'],rtol=1e-10,atol=1e-10)
                med={k:statistics.median(v) for k,v in samples.items()}
                row=dict(modes=n,scenario=scenario,count=count,preparation_ms=dict(frozen=old_prep*1000,candidate=new_prep*1000),
                         samples_ms=samples,median_ms=med,speedup_cold=med['frozen']/med['candidate_cold'],
                         speedup_hint=med['frozen']/med['candidate_hint'],checksums=checks)
                rows.append(row);print(json.dumps({k:row[k] for k in ('modes','scenario','count','median_ms','speedup_hint')}),flush=True)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,default=ROOT/'contact_history.py')
    parser.add_argument('--sizes',type=int,nargs='+',default=[100,1000,10000])
    parser.add_argument('--scenarios',nargs='+',choices=['stick','coherent_slide','reversing_slide','coherent_upper','coherent_mixed','coupled_face'],
                        default=['stick','coherent_slide','reversing_slide','coherent_upper','coherent_mixed','coupled_face'])
    args=parser.parse_args()
    args.candidate=args.candidate.resolve()
    if args.candidate!=ROOT/'contact_history.py':
        candidate_spec=importlib.util.spec_from_file_location('selected_history',args.candidate)
        candidate_module=importlib.util.module_from_spec(candidate_spec);sys.modules[candidate_spec.name]=candidate_module
        candidate_spec.loader.exec_module(candidate_module);ContactHistory=candidate_module.ContactHistory
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,
        hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
                [('frozen',Path(__file__).with_name('frozen_contact_history.py')),('candidate',args.candidate),('harness',Path(__file__))]},
        scope='Python local history updates, validation and energy gates included; preparation/world/detection/group solving excluded')
    receipt['verification']=verification()
    (args.output/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt['verification']),flush=True)
    receipt['benchmarks']=benchmark(args.sizes,args.scenarios)
    (args.output/'results.json').write_text(json.dumps(receipt,indent=2)+'\n')
