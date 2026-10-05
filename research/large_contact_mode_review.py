"""Retain feasible numerical restarts on a stubborn frozen contact cluster."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from research.coulomb_trust_diagnostic import equations
from research.large_contact_completion_review import original_gate


def run():
    path=Path('research/hull-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_0.json');data=json.loads(path.read_text())
    A=np.array(data['A']);b=np.array(data['b']);dep=np.array(data['dependencies']);hi=np.array(data['hi']);warm=np.array(data['p']);normals=np.flatnonzero(dep<0)
    active=normals[warm[normals]>1e-9];selected=np.array(sorted([int(i) for k in active for i in (k,*np.flatnonzero(dep==k))]));mapping={old:new for new,old in enumerate(selected)}
    local=dict(data,A=A[np.ix_(selected,selected)].tolist(),b=b[selected].tolist(),hi=hi[selected].tolist(),lo=np.array(data['lo'])[selected].tolist(),dependencies=[-1 if dep[i]<0 else mapping[dep[i]] for i in selected])
    calc,error=equations(local);ks=np.flatnonzero(np.array(local['dependencies'])<0);N=np.array(local['A']);B=np.array(local['b']);rn=1/N[ks,ks]
    def fb(p,jac=False):
        result=calc(p,jac);u=p[ks]/rn;w=N[ks]@p-B[ks];length=np.hypot(u,w)
        if jac:
            ca=1-np.divide(u,length,out=np.zeros_like(u),where=length>0);cb=1-np.divide(w,length,out=np.zeros_like(w),where=length>0)
            result[ks]=cb[:,None]*N[ks];result[ks,ks]+=ca/rn
        else:result[ks]=u+w-length
        return result
    failed=json.loads(Path('research/completion-large-contact-review/reference_0-fb-trf-normal-qp-10-subset.json').read_text());base=np.array(failed['p']);wbase=A@base-b
    cluster=[30,31,32];total=float(base[cluster].sum());fractions=[(1,0,0),(0,1,0),(0,0,1),(.5,.5,0),(.5,0,.5),(0,.5,.5),(1/3,1/3,1/3),(.1,.8,.1),(.8,.1,.1),(.1,.1,.8)]
    results=[]
    for proportions in fractions:
        initial=base.copy()
        for k,fraction in zip(cluster,proportions):
            initial[k]=fraction*total;t=np.flatnonzero(dep==k);slip=np.linalg.norm(wbase[t]);cap=hi[t[0]]*initial[k]
            initial[t]=-cap*wbase[t]/slip if slip>0 else 0
        started=time.perf_counter();optimized=least_squares(fb,initial[selected],jac=lambda p:fb(p,True),method='trf',max_nfev=1000,ftol=1e-14,xtol=1e-14,gtol=1e-14)
        p=np.zeros(len(b));p[selected]=optimized.x
        for k in normals:
            if p[k]<0 and p[k]>=-data['tolerance_m_s']/A[k,k]:p[k]=0
        result=dict(cluster=cluster,initial_proportions=proportions,initial_p=initial.tolist(),p=p.tolist(),w=(A@p-b).tolist(),
                    nfev=int(optimized.nfev),optimizer_success=bool(optimized.success),elapsed_s=time.perf_counter()-started,**original_gate(data,p))
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k not in ('initial_p','p','w')}),flush=True)
        if result['original_equations_accepted']:break
    package=dict(schema='large-contact-pressure-mode-trials-v1',capture=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 original_coefficient=hi[np.flatnonzero(dep>=0)[0]],max_nfev_per_attempt=1000,trial_count=len(results),trials=results,
                 interpretation='Cone-feasible pressure/tangent initializations are numerical trials only; complete original-system law and energy gate accept the final candidate.')
    Path('research/completion-large-contact-review/reference_0-pressure-mode-trials.json').write_text(json.dumps(package,indent=2)+'\n')


if __name__=='__main__':run()
