"""Independent analysis/solve of frozen 3D circular Coulomb systems.

No new compliance, restitution, contact discovery or friction law is introduced.
A solver's success flag never accepts an impulse: the circular contact residual
and same frozen-system passivity bound are checked separately.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares,linprog


@dataclass(frozen=True)
class System:
    A: np.ndarray
    b: np.ndarray
    contacts: tuple
    upper: np.ndarray

    @classmethod
    def from_dump(cls,data):
        A=np.asarray(data['A'],dtype=float);b=np.asarray(data['b'],dtype=float)
        lo=np.asarray(data['lo'],dtype=float);hi=np.asarray(data['hi'],dtype=float)
        dep=np.asarray(data.get('dep',data.get('dependencies')),dtype=int);n=len(b)
        if A.shape!=(n,n) or lo.shape!=(n,) or hi.shape!=(n,) or dep.shape!=(n,):raise ValueError('incompatible matrix, bounds and dependencies')
        if not np.isfinite(A).all() or not np.isfinite(b).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():raise ValueError('finite data required')
        if not np.allclose(A,A.T,rtol=1e-12,atol=1e-12):raise ValueError('symmetric mobility required')
        contacts=[]
        for k in np.flatnonzero(dep<0):
            tangents=np.flatnonzero(dep==k)
            if lo[k]!=0 or hi[k]<=0 or A[k,k]<=0 or len(tangents)!=2:raise ValueError('one unilateral normal and two tangent rows required')
            if not np.array_equal(lo[tangents],-hi[tangents]) or hi[tangents[0]]!=hi[tangents[1]] or hi[tangents[0]]<0:raise ValueError('one isotropic coefficient required')
            eigen=float(np.linalg.eigvalsh(A[np.ix_(tangents,tangents)])[-1])
            if eigen<=0:raise ValueError('positive tangent mobility required')
            contacts.append((int(k),tuple(int(i) for i in tangents),float(hi[tangents[0]]),1/A[k,k],1/eigen))
        covered=[i for k,ts,*_ in contacts for i in (k,*ts)]
        if sorted(covered)!=list(range(n)):raise ValueError('all rows must belong to a supported contact')
        return cls(A,b,tuple(contacts),hi)

    def equations(self,p,jacobian=False):
        p=np.asarray(p,dtype=float);w=self.A@p-self.b
        F=np.empty_like(p);J=np.zeros_like(self.A) if jacobian else None
        for k,ts,mu,rn,rt in self.contacts:
            t=np.asarray(ts);zn=p[k]-rn*w[k]
            F[k]=(p[k]-max(0.,zn))/rn
            if jacobian:
                if zn>0:J[k]=self.A[k]
                else:J[k,k]=1/rn
            z=p[t]-rt*w[t];length=np.linalg.norm(z);cap=mu*max(0.,p[k])
            if length<=cap and cap>0:
                F[t]=w[t]
                if jacobian:J[t]=self.A[t]
            else:
                direction=z/length if length>0 else np.zeros(2)
                F[t]=(p[t]-cap*direction)/rt
                if jacobian:
                    D=cap/length*(np.eye(2)-np.outer(direction,direction)) if length>0 else np.zeros((2,2))
                    J[t]=D@self.A[t]
                    J[np.ix_(t,t)]+=(np.eye(2)-D)/rt
                    if p[k]>0:J[t,k]-=mu*direction/rt
        return (F,J) if jacobian else F

    def residual(self,p):
        F=self.equations(p)
        return max((max(abs(F[k]),np.linalg.norm(F[list(ts)])) for k,ts,*_ in self.contacts),default=0.)

    def gate(self,p,tolerance):
        p=np.asarray(p);w=self.A@p-self.b
        error=float(self.residual(p))
        change=float(.5*np.dot(p,self.A@p)-self.b@p)
        scale=1+float(np.sum(abs(p*self.b)))
        upper_ok=all(p[k]<=self.upper[k] for k,*_ in self.contacts)
        return dict(accepted=bool(np.isfinite(p).all() and error<=tolerance and change<=tolerance*scale and upper_ok),
                    residual_m_s=error,passive_change_bound_J=change,passivity_scale=scale,
                    upper_bounds_ok=upper_ok,normal_min=float(min((p[k] for k,*_ in self.contacts),default=0.)))


def solve(system,initial=None,tolerance=1e-8,max_newton=64,max_nfev=3000):
    """SVD Newton increments and disclosed trust-region least-squares fallback.

    Intermediate trial impulses are unconstrained; only the exact projected
    equations and passivity gate accept a final solution. Minimum-norm singular
    Newton increments preserve the current gauge instead of forcing arbitrary
    absolute impulse coordinates to zero. No diagonal regularization is added
    to the physical mobility matrix.
    """
    if tolerance<=0 or not np.isfinite(tolerance):raise ValueError('positive finite tolerance required')
    p=np.zeros(len(system.b)) if initial is None else np.asarray(initial,dtype=float).copy()
    if p.shape!=system.b.shape or not np.isfinite(p).all():raise ValueError('finite impulse per row required')
    path=[];ranks=[];method='semismooth-svd';evaluations=0
    for iteration in range(max_newton):
        gate=system.gate(p,tolerance);path.append(gate['residual_m_s'])
        if gate['accepted']:break
        F,J=system.equations(p,True)
        direction,_,rank,_=lstsq(J,-F,cond=1e-13,lapack_driver='gelsd');ranks.append(int(rank))
        merit=float(F@F);accepted=False
        for line in range(30):
            alpha=.5**line;trial=p+alpha*direction;trial_F=system.equations(trial);evaluations+=1
            if np.dot(trial_F,trial_F)<=(1-1e-4*alpha)*merit:
                p=trial;accepted=True;break
        if not accepted:break
    gate=system.gate(p,tolerance)
    optimizer_success=None
    if not gate['accepted'] and max_nfev>0:
        method='semismooth-svd+least-squares'
        # J singularity changes numerical effort, not the material matrix A.
        optimized=least_squares(system.equations,p,jac=lambda value:system.equations(value,True)[1],
                                method='trf',tr_solver='exact',xtol=1e-13,ftol=1e-13,gtol=1e-13,
                                max_nfev=max_nfev,x_scale='jac')
        p=optimized.x;optimizer_success=bool(optimized.success);evaluations+=optimized.nfev
        gate=system.gate(p,tolerance)
    return dict(impulse=p,velocity=system.A@p-system.b,method=method,newton_residual_path=path,
                jacobian_ranks=ranks,evaluations=evaluations,optimizer_success=optimizer_success,**gate)


def gauge_candidates(system,p,tolerance=1e-8,max_directions=4):
    """Move to neighboring friction faces through a mechanical impulse gauge.

    Candidate directions must lie in the numerical Jacobian nullspace AND have
    a certified negligible mobility response. Both signs are examined and stop
    at the first unilateral-normal or circular-friction boundary. This is an
    initialization change, not compliance or a different friction capacity.
    """
    p=np.asarray(p);_,J=system.equations(p,True);_,singular,V=np.linalg.svd(J)
    threshold=1e-12*max(1.,singular[0] if len(singular) else 0.)
    rank=int(np.sum(singular>threshold));out=[]
    for index,direction in enumerate(V[rank:rank+max_directions]):
        direction=direction.copy()
        pivot=np.argmax(abs(direction))
        if direction[pivot]<0:direction=-direction
        response=system.A@direction
        if np.linalg.norm(response,ord=np.inf)>1e-12*max(1.,np.linalg.norm(system.A,ord=np.inf)):continue
        for sign in (1.,-1.):
            d=sign*direction;limits=[];blocked=False
            for k,ts,mu,*_ in system.contacts:
                t=list(ts)
                if np.linalg.norm(d[[k,*t]])<1e-10:continue
                if d[k]<0:
                    boundary=-p[k]/d[k]
                    if boundary<=1e-10:blocked=True;break
                    limits.append((boundary,k,'normal'))
                a=float(d[t]@d[t]-mu**2*d[k]**2)
                b=float(2*(p[t]@d[t]-mu**2*p[k]*d[k]))
                c=float(p[t]@p[t]-mu**2*p[k]**2)
                if abs(c)<1e-12*max(1.,p[t]@p[t]):c=0.
                if c==0 and b>1e-12:blocked=True;break
                roots=np.roots([a,b,c]) if abs(a)>1e-20 else ([-c/b] if abs(b)>1e-20 else [])
                for root in roots:
                    if np.isreal(root) and root>1e-10:
                        value=float(root)
                        # A tangency that does not leave the cone is not a
                        # neighboring mode boundary along this direction.
                        if 2*a*value+b>=-1e-12:limits.append((value,k,'tangent'))
            if blocked or not limits:continue
            alpha,k,kind=min(limits,key=lambda item:item[0]);trial=p+alpha*d
            velocity_change=float(np.linalg.norm(system.A@(trial-p),ord=np.inf))
            if velocity_change>tolerance*.01:continue
            feasible=all(trial[n]>=-1e-10 and np.linalg.norm(trial[list(t)])<=mu*max(trial[n],0.)+1e-10
                         for n,t,mu,*_ in system.contacts)
            if not feasible:continue
            out.append(dict(impulse=trial,certificate=dict(direction=direction.tolist(),sign=sign,
                            alpha=alpha,boundary_normal_row=k,boundary_kind=kind,
                            mobility_null_residual=float(np.linalg.norm(response,ord=np.inf)),
                            jacobian_null_residual=float(np.linalg.norm(J@direction,ord=np.inf)),
                            velocity_change_m_s=velocity_change,
                            original_residual_m_s=float(system.residual(p)),
                            relocated_residual_m_s=float(system.residual(trial)))))
    return out


def recover(system,initial=None,tolerance=1e-8,max_newton=64,max_nfev=3000):
    """Deterministic face exploration, then cold Newton/least-squares recovery.

    The final acceptance gate is unchanged. A neutral pressure redistribution
    is permitted as a restart because strict merit decrease inside the current
    inconsistent sticking face can otherwise make a valid adjacent face
    unreachable. Failure remains an explicit unaccepted result.
    """
    original=np.zeros(len(system.b)) if initial is None else np.asarray(initial,dtype=float)
    attempts=[]
    def attempt(value,label,least_squares=0):
        result=solve(system,value,tolerance,max_newton=max_newton,max_nfev=least_squares)
        attempts.append(dict(label=label,accepted=result['accepted'],residual_m_s=result['residual_m_s'],
                             newton_steps=len(result['jacobian_ranks']),evaluations=result['evaluations']))
        return result
    result=attempt(original,'warm-newton')
    if result['accepted']:result['recovery_attempts']=attempts;return result
    for start_label,start in [('warm',original),('stagnated',result['impulse'])]:
        for candidate in gauge_candidates(system,start,tolerance):
            trial=attempt(candidate['impulse'],start_label+'-mechanical-null-gauge')
            if trial['accepted']:
                trial['method']='semismooth-svd+mechanical-null-gauge'
                trial['gauge_certificate']=candidate['certificate'];trial['recovery_attempts']=attempts
                return trial
    result=attempt(np.zeros_like(original),'cold-newton')
    if not result['accepted'] and max_nfev>0:result=attempt(result['impulse'],'least-squares',max_nfev)
    if result['accepted'] and attempts[-1]['label']=='cold-newton':result['method']='semismooth-svd+cold-restart'
    result['recovery_attempts']=attempts
    return result


def analyze(system,impulse=None,tolerance=1e-8):
    eigen=np.linalg.eigvalsh(system.A);singular=np.linalg.svd(system.A,compute_uv=False)
    relative_cutoff=1e-12*max(1.,singular[0] if len(singular) else 0.)
    k=np.array([contact[0] for contact in system.contacts]);N=system.A[k]
    # A normalized nonnegative left-null combination can expose contradictory
    # normal RHS targets even when all impulse coordinates are unrestricted.
    certificate=linprog(-system.b[k],A_eq=np.vstack([N.T,np.ones(len(k))]),
                        b_eq=np.r_[np.zeros(len(system.b)),1.],bounds=(0,None),method='highs') if len(k) else None
    witness=None
    if certificate is not None and certificate.success:
        weights=certificate.x;null_error=float(np.linalg.norm(weights@N,ord=np.inf))
        target=float(weights@system.b[k])
        if target>tolerance:
            witness=dict(weights=weights.tolist(),weighted_normal_target_m_s=target,
                         left_null_residual=null_error,
                         interpretation='Numerical left-null target incompatibility; exact infeasibility only if the weighted mobility row is exactly zero.')
    report=dict(rows=len(system.b),contacts=len(system.contacts),rank=int(np.sum(singular>relative_cutoff)),
                min_eigenvalue=float(eigen[0]) if len(eigen) else 0.,max_eigenvalue=float(eigen[-1]) if len(eigen) else 0.,
                singular_values=singular.tolist(),normal_target_witness=witness)
    if impulse is not None:
        p=np.asarray(impulse);F,J=system.equations(p,True);w=system.A@p-system.b
        active=np.array([c[0] for c in system.contacts if p[c[0]]>1e-10])
        report['iterate']=system.gate(p,tolerance)
        report['jacobian_rank']=int(np.linalg.matrix_rank(J,tol=1e-12*max(1.,np.linalg.norm(J,2))))
        report['active_normal_rows']=active.tolist()
        if len(active):
            target=system.b[active];Aactive=system.A[active]
            fitted=Aactive@np.linalg.lstsq(Aactive,target,rcond=1e-12)[0]
            report['active_normal_target_range_error_m_s']=float(np.linalg.norm(fitted-target,ord=np.inf))
        report['worst_contacts']=sorted([dict(normal=int(k),tangents=list(ts),friction=mu,
                            normal_impulse=float(p[k]),normal_velocity=float(w[k]),
                            tangential_impulse_norm=float(np.linalg.norm(p[list(ts)])),
                            tangential_velocity_norm=float(np.linalg.norm(w[list(ts)])),
                            normal_residual=float(abs(F[k])),tangent_residual=float(np.linalg.norm(F[list(ts)])))
                            for k,ts,mu,*_ in system.contacts],key=lambda c:max(c['normal_residual'],c['tangent_residual']),reverse=True)[:10]
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('dump');parser.add_argument('--output');parser.add_argument('--max-nfev',type=int,default=3000)
    args=parser.parse_args();data=json.loads(Path(args.dump).read_text());system=System.from_dump(data)
    tolerance=float(data.get('tolerance_m_s',data.get('tolerance',1e-8)));initial=data.get('p',data.get('x'))
    report=dict(native=analyze(system,initial,tolerance))
    result=recover(system,initial,tolerance,max_nfev=args.max_nfev)
    report['independent_solve']={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}
    report['solved_analysis']=analyze(system,result['impulse'],tolerance)
    text=json.dumps(report,indent=2)+'\n'
    if args.output:Path(args.output).write_text(text)
    print(json.dumps(dict(rows=len(system.b),native_residual=report['native'].get('iterate',{}).get('residual_m_s'),
                         independent_residual=result['residual_m_s'],accepted=result['accepted'],method=result['method']),indent=2))


if __name__=='__main__':main()
