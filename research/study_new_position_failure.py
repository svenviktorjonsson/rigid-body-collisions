"""Independent feasible-normal diagnosis for the later 291-row pose capture."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,minimize,least_squares
from research.audit_large_contact_completion import check


def run():
    path=Path('research/hull-active-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_0.json');d=json.loads(path.read_text())
    A=np.array(d['A']);b=np.array(d['b']);dep=np.array(d['dependencies']);ns=np.flatnonzero(dep<0);M=A[np.ix_(ns,ns)];rhs=b[ns]
    assert d['phase']=='position' and np.all(np.array(d['hi'])[dep>=0]==0)
    t=time.perf_counter();lp=linprog(np.ones(len(ns)),A_ub=-M,b_ub=-rhs,bounds=[(0,None)]*len(ns),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    history=[];out=Path('research/new-hull-contact-review');out.mkdir(exist_ok=True)
    package=dict(schema='later-normal-position-independent-trials-v1',capture=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),normal_rows=ns.tolist(),
                 normal_lp_success=bool(lp.success),normal_lp_status=lp.message,minimum_lp_normal_slack=float(np.min(M@lp.x-rhs)) if lp.success else None,trials=history)
    def archive():(out/'fast_rotate_shake27_hulls7301-reference_0-position.json').write_text(json.dumps(package,indent=2)+'\n')
    scale=max(float(np.max(rhs*rhs/np.diag(M))),1e-30)
    for objective_scale in (1.,1/scale):
        r=minimize(lambda x:objective_scale*(.5*x@M@x-rhs@x),np.zeros(len(ns)),jac=lambda x:objective_scale*(M@x-rhs),bounds=[(0,None)]*len(ns),method='SLSQP',options={'ftol':1e-15,'maxiter':3000})
        p=np.zeros(len(b));p[ns]=r.x;gate=check(path,p);history.append(dict(method='normal-SLSQP',objective_scale=objective_scale,nit=r.nit,optimizer_success=bool(r.success),original_gate=gate));archive();print(objective_scale,r.nit,gate['independent_full_original_residual_m_s'],gate['accepted'],flush=True)
        if gate['accepted']:return
        rho=1/np.diag(M)
        def fb(x,jac=False):
            u=x/rho;w=M@x-rhs;length=np.hypot(u,w)
            if not jac:return u+w-length
            ca=1-np.divide(u,length,out=np.zeros_like(u),where=length>0);cb=1-np.divide(w,length,out=np.zeros_like(w),where=length>0)
            J=cb[:,None]*M;J[np.arange(len(ns)),np.arange(len(ns))]+=ca/rho;return J
        opt=least_squares(fb,r.x,jac=lambda x:fb(x,True),max_nfev=500,ftol=1e-14,xtol=1e-14,gtol=1e-14)
        p=np.zeros(len(b));p[ns]=opt.x;gate=check(path,p);history.append(dict(method='normal-FB-TRF',start='SLSQP-'+str(objective_scale),nfev=opt.nfev,optimizer_success=bool(opt.success),original_gate=gate));archive();print('FB',opt.nfev,gate['independent_full_original_residual_m_s'],gate['accepted'],flush=True)
        if gate['accepted']:return


if __name__=='__main__':run()
