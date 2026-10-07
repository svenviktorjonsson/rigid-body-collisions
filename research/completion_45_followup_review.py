"""Reproduce bounded, mostly unsuccessful 45-row numerical face experiments.

No trial modifies the engine, physical A, friction coefficient, or final gate.
Use --experiment to select an archived experiment. These outputs retain failed
attempts: solver success flags and passive energy alone are not acceptance.
"""
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from research.coulomb_trust_diagnostic import equations
from research.coulomb_diagnostics import System,solve,gauge_candidates
from research.large_contact_completion_review import original_gate


def run(experiment):
    cap=json.loads(Path('research/hull-completion/results/rejections/fast_shake8_hulls42/reference_1.json').read_text())
    root=Path('research/completion-large-contact-review');pack=json.loads((root/'seed42-reference_1-sticking-trials.json').read_text())
    sel=np.array(pack['selected_rows']);dep=np.array(cap['dependencies']);A=np.array(cap['A']);b=np.array(cap['b']);hi=np.array(cap['hi']);normal=np.flatnonzero(dep<0)
    def reduced(rows):
        mapping={int(k):i for i,k in enumerate(rows)}
        return dict(cap,A=A[np.ix_(rows,rows)].tolist(),b=b[rows].tolist(),hi=hi[rows].tolist(),lo=np.array(cap['lo'])[rows].tolist(),dependencies=[-1 if dep[k]<0 else mapping[int(dep[k])] for k in rows])
    local=reduced(sel);f,error=equations(local);system=System.from_dump(local);out=[]
    def record(q,rows,started,**details):
        full=np.zeros(len(b));full[rows]=q
        if experiment=='normal-release':full[normal]=np.where((full[normal]<0)&(full[normal]>=-1e-12),0,full[normal])
        row=dict(**details,p=full.tolist(),elapsed_s=time.perf_counter()-started,**original_gate(cap,full));out.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('p','preconditioner','initial_p')}),flush=True)
        (root/('seed42-reference_1-'+experiment+'.json')).write_text(json.dumps(out,indent=2)+'\n')
        return row
    def optimize(fun,jac,p,budget=500):
        return least_squares(fun,p,jac=jac,max_nfev=budget,ftol=1e-14,xtol=1e-14,gtol=1e-14)
    starts=[('warm',np.array(cap['p'])),('trf500',np.array(pack['trials'][0]['p']))]
    if experiment=='minnorm-newton':
        for label,p in starts:
            t=time.perf_counter();r=solve(system,p[sel],tolerance=1e-12,max_newton=100,max_nfev=0);q=r.pop('impulse');r.pop('velocity')
            row=dict(start=label,p=np.zeros(len(b)),elapsed_s=time.perf_counter()-t,**r);row['p'][sel]=q;row['full_gate']=original_gate(cap,row['p']);row['p']=row['p'].tolist();out.append(row)
        (root/'seed42-reference_1-minnorm-newton.json').write_text(json.dumps(out,indent=2)+'\n')
    elif experiment=='neutral-gauges':
        for label,p in starts:
            for c in gauge_candidates(system,p[sel]):
                t=time.perf_counter();r=optimize(f,lambda z:f(z,True),c['impulse']);record(r.x,sel,t,start=label,certificate=c['certificate'],initial_p=c['impulse'].tolist())
    elif experiment=='left-preconditioned':
        for label,p in starts:
            U,S,V=np.linalg.svd(f(p[sel],True));mask=S>1e-10;W=(U[:,mask]/S[mask]).T
            for boost in (.001,.01,.1,1.):
                Q=W.copy();Q[-2:]*=boost;t=time.perf_counter();r=optimize(lambda z:Q@f(z),lambda z:Q@f(z,True),p[sel])
                record(r.x,sel,t,start=label,boost=boost,nfev=r.nfev,optimizer_success=bool(r.success),singular_values=S.tolist(),preconditioner=Q.tolist())
    elif experiment=='invertible-preconditioned':
        p=starts[0][1][sel];U,S,V=np.linalg.svd(f(p,True))
        for floor in (1e-4,1e-5,1e-6,1e-7,1e-8,1e-9):
            Q=(U/np.maximum(S,floor)).T;t=time.perf_counter();r=optimize(lambda z:Q@f(z),lambda z:Q@f(z,True),p)
            record(r.x,sel,t,floor=floor,nfev=r.nfev,optimizer_success=bool(r.success),preconditioner=Q.tolist())
    elif experiment=='normal-release':
        active=normal[np.array(cap['p'])[normal]>1e-9]
        for count in range(1,4):
            for removed in itertools.combinations((3,4,5,6),count):
                rows=np.array(sorted([int(i) for k in active if k not in removed for i in (k,*np.flatnonzero(dep==k))]));calc,error=equations(reduced(rows))
                t=time.perf_counter();r=optimize(calc,lambda z:calc(z,True),np.array(cap['p'])[rows],250);row=record(r.x,rows,t,removed=removed,nfev=r.nfev)
                if row['original_equations_accepted']:return


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True,choices=['minnorm-newton','neutral-gauges','left-preconditioned','invertible-preconditioned','normal-release']);run(p.parse_args().experiment)
