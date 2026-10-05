"""Independent exact-law diagnosis of later frozen production rejections."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,minimize
from research.coulomb_trust_diagnostic import equations
from research.audit_large_contact_completion import check


def run(path):
    data=json.loads(path.read_text());A=np.array(data['A']);b=np.array(data['b']);dep=np.array(data['dependencies']);hi=np.array(data['hi']);ns=np.flatnonzero(dep<0);rn=1/A[ns,ns]
    out=Path('research/new-hull-contact-review');out.mkdir(exist_ok=True);history=[]
    def archive():
        (out/(path.parent.name+'-'+path.stem+'.json')).write_text(json.dumps(dict(schema='later-frozen-contact-independent-trials-v1',capture=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),trials=history),indent=2)+'\n')
    def optimize(p,alpha,label,budget=250):
        trial=dict(data);bounds=hi.copy();bounds[dep>=0]*=alpha;trial['hi']=bounds.tolist();f,error=equations(trial)
        def fb(p,jac=False):
            result=f(p,jac);u=p[ns]/rn;w=A[ns]@p-b[ns];length=np.hypot(u,w)
            if jac:
                ca=1-np.divide(u,length,out=np.zeros_like(u),where=length>0);cb=1-np.divide(w,length,out=np.zeros_like(w),where=length>0)
                result[ns]=cb[:,None]*A[ns];result[ns,ns]+=ca/rn
            else:result[ns]=u+w-length
            return result
        t=time.perf_counter();r=least_squares(fb,p,jac=lambda z:fb(z,True),max_nfev=budget,ftol=1e-14,xtol=1e-14,gtol=1e-14)
        gate=check(path,r.x);row=dict(label=label,alpha=alpha,initial_p=p.tolist(),raw_p=r.x.tolist(),nfev=r.nfev,optimizer_success=bool(r.success),elapsed_s=time.perf_counter()-t,trial_residual_m_s=float(error(r.x)),original_gate=gate)
        history.append(row);archive();print(label,alpha,r.nfev,gate['independent_full_original_residual_m_s'],gate['accepted'],flush=True)
        return r.x,gate
    warm=np.array(data['p']);p,gate=optimize(warm,1.,'direct-warm')
    if gate['accepted']:return
    q=minimize(lambda x:.5*x@A[np.ix_(ns,ns)]@x-b[ns]@x,np.zeros(len(ns)),jac=lambda x:A[np.ix_(ns,ns)]@x-b[ns],bounds=[(0,None)]*len(ns),method='SLSQP',options={'ftol':1e-15,'maxiter':2000})
    p=np.zeros(len(b));p[ns]=q.x
    for alpha in np.linspace(0,1,41):p,gate=optimize(p,float(alpha),'continuation-normal-qp')
    if gate['accepted']:return
    base=p.copy();w=A@base-b;f,error=equations(data);F=f(base)
    ranking=sorted(ns,key=lambda k:np.linalg.norm(F[np.flatnonzero(dep==k)]),reverse=True)
    for k in ranking[:8]:
        ts=np.flatnonzero(dep==k)
        if base[k]<=0 or np.linalg.norm(w[ts])<=0:continue
        initial=base.copy();initial[ts]=-hi[ts[0]]*initial[k]*w[ts]/np.linalg.norm(w[ts]);p,gate=optimize(initial,1.,'opposing-slip-'+str(k))
        if gate['accepted']:return


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);run(p.parse_args().capture)
