"""Independent original-equation evidence for one rejected normal projection."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,minimize
from scipy.linalg import lstsq


def run():
    capture=Path('research/hull-completion/results/rejections/fast_shake8_hulls42/reference_0.json')
    data=json.loads(capture.read_text());A=np.asarray(data['A']);b=np.asarray(data['b']);dep=np.asarray(data['dependencies']);hi=np.asarray(data['hi'])
    normals=np.flatnonzero(dep<0);M=A[np.ix_(normals,normals)];bn=b[normals]
    primal=linprog(np.ones(len(normals)),A_ub=-M,b_ub=-bn,bounds=[(0,None)]*len(normals),method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    certificate=linprog(-bn,A_eq=np.vstack([M.T,np.ones(len(normals))]),b_eq=np.r_[np.zeros(len(normals)),1],
                        bounds=[(0,None)]*len(normals),method='highs')
    optimized=minimize(lambda p:.5*p@M@p-bn@p,np.zeros(len(normals)),jac=lambda p:M@p-bn,
                       bounds=[(0,None)]*len(normals),method='SLSQP',options={'ftol':1e-15,'maxiter':3000})
    p=np.zeros(len(b));p[normals]=optimized.x;w=A@p-b
    errors=[]
    for k in normals:
        t=np.flatnonzero(dep==k);rho=1/np.linalg.eigvalsh(A[np.ix_(t,t)])[-1]
        z=p[t]-rho*w[t];cap=hi[t[0]]*p[k];projection=z*min(1.,cap/np.linalg.norm(z)) if np.linalg.norm(z)>0 else z
        errors.extend((abs(p[k]-max(0,p[k]-w[k]/A[k,k]))*A[k,k],np.linalg.norm(p[t]-projection)/rho))
    residual=float(max(errors));change=float(.5*p@A@p-b@p);scale=float(1+sum(abs(p*b)))
    accepted=bool(np.isfinite(p).all() and residual<=data['tolerance_m_s'] and change<=data['tolerance_m_s']*scale and
                  np.all(p[normals]>=0) and np.all(p[normals]<=hi[normals]))
    assert primal.success and accepted
    package=dict(schema='independent-normal-position-recovery-v1',capture=str(capture),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
                 normal_rows=normals.tolist(),normal_lp_feasible=bool(primal.success),normal_lp_min_slack=float(np.min(M@primal.x-bn)),
                 nonnegative_left_null_certificate_found=bool(certificate.success and -certificate.fun>1e-8),
                 optimizer_success=bool(optimized.success),optimizer_iterations=int(optimized.nit),original_equations_accepted=accepted,
                 residual_m_s=residual,passive_change_bound_J=change,p=p.tolist(),w=w.tolist(),
                 interpretation='Original zero-friction position NCP is feasible; warm normal0 must leave its active face. This is a captured-system diagnostic, not evidence of an accepted full hull trajectory.')
    warm=np.asarray(data['p'])[normals];active=np.flatnonzero(warm>1e-9);attempts=[];exact=None
    for count in range(4):
        for drops in itertools.combinations(range(len(active)),count):
            keep=np.delete(active,list(drops));candidate=np.zeros(len(normals))
            candidate[keep]=lstsq(M[np.ix_(keep,keep)],bn[keep],cond=1e-13)[0]
            response=M@candidate-bn
            error=float(np.max(abs(candidate-np.maximum(0,candidate-response/np.diag(M)))*np.diag(M)))
            feasible=bool(np.min(candidate)>=0)
            attempts.append(dict(released_normal_rows=normals[active[list(drops)]].tolist(),residual_m_s=error,nonnegative=feasible))
            if feasible and error<1e-12:
                exact=dict(**attempts[-1],p_normal=candidate.tolist(),w_normal=response.tolist(),
                           passive_change_bound_J=float(.5*candidate@M@candidate-bn@candidate));break
        if exact is not None:break
    assert exact is not None and exact['passive_change_bound_J']<=0
    package.update(normal_face_attempts=attempts,exact_face_solution=exact,
                   exact_face_interpretation='Bounded candidate active-face release changes only the numerical search; the original unilateral and energy equations independently accept the final normal impulse. Tangents have zero configured capacity and remain zero.')
    out=Path('research/completion-position-review');out.mkdir(exist_ok=True);(out/'position54-recovery.json').write_text(json.dumps(package,indent=2)+'\n')
    print(json.dumps({k:v for k,v in package.items() if k not in ('p','w','normal_rows','normal_face_attempts','exact_face_solution')}))
    print(json.dumps(dict(face_attempts=len(attempts),exact_face_residual_m_s=exact['residual_m_s'],released_normals=exact['released_normal_rows'])))


if __name__=='__main__':run()
