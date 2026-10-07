"""Original-law boundary restarts for the diagnosed 69/70 sticking face.

The frozen material law is unchanged. Guesses are numerical search only; each
returned candidate is clamped only for negligible negative normal roundoff and
checked against all original rows, including excluded contacts.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from research.coulomb_trust_diagnostic import equations
from research.large_contact_completion_review import original_gate


def run():
    path=Path('research/hull-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_0.json');data=json.loads(path.read_text())
    A=np.array(data['A']);b=np.array(data['b']);dep=np.array(data['dependencies']);hi=np.array(data['hi']);normals=np.flatnonzero(dep<0)
    active=normals[np.array(data['p'])[normals]>1e-9];selected=np.array(sorted([int(i) for k in active for i in (k,*np.flatnonzero(dep==k))]));mapping={old:new for new,old in enumerate(selected)}
    local=dict(data,A=A[np.ix_(selected,selected)].tolist(),b=b[selected].tolist(),hi=hi[selected].tolist(),lo=np.array(data['lo'])[selected].tolist(),dependencies=[-1 if dep[i]<0 else mapping[dep[i]] for i in selected])
    calc,error=equations(local);ks=np.flatnonzero(np.array(local['dependencies'])<0);N=np.array(local['A']);B=np.array(local['b']);rn=1/N[ks,ks]
    def fb(p,jac=False):
        result=calc(p,jac);u=p[ks]/rn;w=N[ks]@p-B[ks];length=np.hypot(u,w)
        if jac:
            ca=1-np.divide(u,length,out=np.zeros_like(u),where=length>0);cb=1-np.divide(w,length,out=np.zeros_like(w),where=length>0)
            result[ks]=cb[:,None]*N[ks];result[ks,ks]+=ca/rn
        else:result[ks]=u+w-length
        return result
    source=Path('research/completion-large-contact-review/reference_0-fb-trf-normal-qp-10-subset.json');base=np.array(json.loads(source.read_text())['p']);wbase=A@base-b
    results=[];trials=[]
    for cluster in ([69],[70],[69,70]):
        for sign in (-1,1):trials.append(dict(cluster=cluster,sign=sign,pressure_fraction=None))
    for fraction in (0.,.1,.25,.5,.75,.9,1.):
        for sign in (-1,1):trials.append(dict(cluster=[69,70],sign=sign,pressure_fraction=fraction))
    for spec in trials:
        initial=base.copy()
        if spec['pressure_fraction'] is not None:
            total=base[69]+base[70];initial[69]=total*spec['pressure_fraction'];initial[70]=total*(1-spec['pressure_fraction'])
        for k in spec['cluster']:
            t=np.flatnonzero(dep==k);initial[t]=spec['sign']*hi[t[0]]*initial[k]*wbase[t]/np.linalg.norm(wbase[t])
        started=time.perf_counter();optimized=least_squares(fb,initial[selected],jac=lambda p:fb(p,True),method='trf',max_nfev=1000,ftol=1e-14,xtol=1e-14,gtol=1e-14)
        p=np.zeros(len(b));p[selected]=optimized.x;raw=p.copy()
        p[normals]=np.where((p[normals]<0)&(p[normals]>=-1e-12),0,p[normals])
        result=dict(**spec,initial_p=initial.tolist(),raw_p=raw.tolist(),p=p.tolist(),w=(A@p-b).tolist(),nfev=int(optimized.nfev),optimizer_success=bool(optimized.success),elapsed_s=time.perf_counter()-started,**original_gate(data,p))
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k not in ('initial_p','raw_p','p','w')}),flush=True)
        package=dict(schema='large-contact-sticking-mode-trials-v1',capture=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),base_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),max_nfev_per_attempt=1000,trials=results)
        Path('research/completion-large-contact-review/reference_0-sticking-mode-trials.json').write_text(json.dumps(package,indent=2)+'\n')
        if result['original_equations_accepted']:break


if __name__=='__main__':run()
