"""Frozen-law follow-up exploration; published diagnostic implementation unchanged."""
import itertools
import json
from pathlib import Path
import time
import numpy as np
from research.coulomb_diagnostics import System,solve,gauge_candidates,analyze,recover


def boundary(system,p,d,tolerance=1e-8):
    d=np.asarray(d);length=np.linalg.norm(d)
    if length<1e-12:return []
    d=d/length;out=[]
    for sign in [1.,-1.]:
        direction=sign*d;limits=[];blocked=False
        for k,ts,mu,*_ in system.contacts:
            t=list(ts)
            if np.linalg.norm(direction[[k,*t]])<1e-10:continue
            if direction[k]<0:
                value=-p[k]/direction[k]
                if value<=1e-10:blocked=True;break
                limits.append((value,k,'normal'))
            a=direction[t]@direction[t]-mu**2*direction[k]**2
            b=2*(p[t]@direction[t]-mu**2*p[k]*direction[k])
            c=p[t]@p[t]-mu**2*p[k]**2
            if abs(c)<1e-12*max(1.,p[t]@p[t]):c=0.
            if c==0 and b>1e-12:blocked=True;break
            roots=np.roots([a,b,c]) if abs(a)>1e-20 else ([-c/b] if abs(b)>1e-20 else [])
            for root in roots:
                if np.isreal(root) and root>1e-10 and 2*a*root+b>=-1e-12:limits.append((float(root),k,'tangent'))
        if blocked or not limits:continue
        alpha,k,kind=min(limits,key=lambda item:item[0]);candidate=p+alpha*direction
        change=np.linalg.norm(system.A@(candidate-p),ord=np.inf)
        feasible=all(candidate[k]>=-1e-10 and np.linalg.norm(candidate[list(t)])<=mu*max(candidate[k],0.)+1e-10 for k,t,mu,*_ in system.contacts)
        if feasible and change<=.01*tolerance:
            out.append(dict(impulse=candidate,normal_row=k,boundary_kind=kind,alpha=alpha,sign=sign,
                            direction=d.tolist(),velocity_change_m_s=float(change)))
    return out


def combined_candidates(system,p,tolerance=1e-8):
    _,J=system.equations(p,True);_,s,V=np.linalg.svd(J)
    rank=np.sum(s>1e-12*max(1.,s[0]));B=V[rank:].T
    if B.shape[1]>6:return []
    directions=[]
    for coeff in itertools.product([-1.,0.,1.],repeat=B.shape[1]):
        coeff=np.asarray(coeff)
        nonzero=np.flatnonzero(coeff)
        if len(nonzero) and coeff[nonzero[0]]==1:directions.append(('lattice',B@coeff))
    # Preserve a nearly saturated contact's impulse while redistributing the
    # remaining mechanical pressure gauge between the other contact points.
    for k,t,mu,*_ in system.contacts:
        if mu*p[k]-np.linalg.norm(p[list(t)])>1e-4:continue
        constraint=B[list(t)];_,s2,V2=np.linalg.svd(constraint,full_matrices=True)
        rank2=np.sum(s2>1e-12*max(1.,s2[0] if len(s2) else 0.))
        for coeff in V2[rank2:]:directions.append(('preserve-contact-'+str(k),B@coeff))
    out=[];seen=set()
    for label,d in directions:
        if np.linalg.norm(system.A@d,ord=np.inf)>1e-11:continue
        for candidate in boundary(system,p,d,tolerance):
            key=tuple(np.round(candidate['impulse'],9))
            if key not in seen:candidate['direction_kind']=label;out.append(candidate);seen.add(key)
    return out


def targeted_slip_candidates(system,p,tolerance=1e-8):
    """Neutral relocation directly enforces one contact's opposing traction.

    Within a mechanical nullspace the velocities do not change. Consequently
    the slip direction is fixed, and one contact's saturated-opposing-traction
    condition is linear in the null coefficients, including its pressure cap.
    The remaining one-dimensional freedom is intersected with every circular
    cone. Feasible endpoints and their whole straight path remain in the convex
    impulse cone, so no intermediate force-budget violation is required.
    """
    _,J=system.equations(p,True);_,s,V=np.linalg.svd(J)
    rank=np.sum(s>1e-12*max(1.,s[0]));B=V[rank:].T
    if not B.shape[1] or np.max(abs(system.A@B))>1e-11:return []
    w=system.A@p-system.b;out=[]
    for k,ts,mu,*_ in system.contacts:
        ts=list(ts);speed=np.linalg.norm(w[ts])
        if speed<=tolerance or mu==0 or p[k]<=0:continue
        direction=w[ts]/speed
        E=B[ts]+mu*np.outer(direction,B[k]);rhs=-p[ts]-mu*p[k]*direction
        coeff=np.linalg.lstsq(E,rhs,rcond=1e-12)[0]
        if np.linalg.norm(E@coeff-rhs)>1e-10:continue
        _,s2,V2=np.linalg.svd(E,full_matrices=True);rank2=np.sum(s2>1e-12*max(1.,s2[0]))
        free=V2[rank2:]
        if len(free)>1:continue
        base=p+B@coeff;d=B@free[0] if len(free) else np.zeros(len(p))
        critical=[0.]
        for n,tt,friction,*_ in system.contacts:
            tt=list(tt)
            if abs(d[n])>1e-20:critical.append(-base[n]/d[n])
            a=d[tt]@d[tt]-friction**2*d[n]**2
            b=2*(base[tt]@d[tt]-friction**2*base[n]*d[n])
            c=base[tt]@base[tt]-friction**2*base[n]**2
            roots=np.roots([a,b,c]) if abs(a)>1e-20 else ([-c/b] if abs(b)>1e-20 else [])
            critical.extend(float(root) for root in roots if np.isreal(root) and np.isfinite(root))
        critical=sorted(set(critical));values=critical+[(a+b)/2 for a,b in zip(critical,critical[1:])]
        values.sort(key=abs)
        for value in values:
            candidate=base+value*d
            feasible=all(candidate[n]>=-1e-10 and np.linalg.norm(candidate[list(t)])<=friction*max(candidate[n],0.)+1e-10
                         for n,t,friction,*_ in system.contacts)
            change=float(np.linalg.norm(system.A@(candidate-p),ord=np.inf))
            if feasible and change<=.01*tolerance:
                out.append(dict(impulse=candidate,direction_kind='target-opposing-slip',normal_row=k,
                                velocity_change_m_s=change,free_coordinate=float(value),
                                target_velocity_m_s=w[ts].tolist(),coefficient=coeff.tolist()))
                break
    return out


def explore(system,initial,tolerance=1e-8,max_nodes=80,max_depth=3):
    initial=np.asarray(initial);queue=[(initial,0,[])];visited=set();attempts=[]
    for node in range(max_nodes):
        if not queue:break
        p,depth,history=queue.pop(0)
        if depth>=max_depth:continue
        for candidate in targeted_slip_candidates(system,p,tolerance)+combined_candidates(system,p,tolerance):
            key=tuple(np.round(candidate['impulse'],5))
            if key in visited:continue
            visited.add(key)
            result=solve(system,candidate['impulse'],tolerance,max_newton=16,max_nfev=0)
            record={k:v for k,v in candidate.items() if k!='impulse'}
            record.update(accepted=result['accepted'],residual_m_s=result['residual_m_s'],depth=depth+1)
            attempts.append(record)
            if result['accepted']:return dict(result=result,attempts=attempts,history=history+[record])
            queue.append((result['impulse'],depth+1,history+[record]))
            if len(attempts)>=max_nodes:return dict(result=result,attempts=attempts,history=None)
    return dict(result=None,attempts=attempts,history=None)


def opposing_slip_restart(system,initial,tolerance=1e-8,max_calls=256,max_restarts=8):
    """Bounded, basis-invariant face restart; physical law remains unchanged.

    A stalled least-squares merit can have nonzero tangential residual on an
    inconsistent sticking face. Reset one such contact to its existing normal
    pressure cap, opposing its current post-slip velocity, then solve the same
    coupled projection equations. This changes a numerical initialization,
    not an applied impulse episode; unlike mechanical gauge exploration its
    intermediate contact velocities need not remain unchanged. Final circular
    law/passivity gates, not the starting cone or optimizer status, accept.
    """
    original=np.asarray(initial,dtype=float)
    warm=solve(system,original,tolerance,max_newton=min(64,max_calls),max_nfev=0)
    used=len(warm['jacobian_ranks']);attempts=[]
    if warm['accepted']:return dict(result=warm,attempts=attempts,newton_calls=used)
    p=np.asarray(warm['impulse']);w=system.A@p-system.b
    candidates=[]
    F=system.equations(p)
    for k,ts,mu,*_ in system.contacts:
        t=list(ts);speed=np.linalg.norm(w[t]);residual=np.linalg.norm(F[t])
        if speed>tolerance and residual>tolerance and p[k]>0 and mu>0:
            candidates.append((float(residual),k,t,mu))
    candidates.sort(key=lambda item:(-item[0],item[1]))
    for _,k,t,mu in candidates[:max_restarts]:
        if used>=max_calls:break
        q=p.copy();q[t]=-mu*p[k]*w[t]/np.linalg.norm(w[t])
        result=solve(system,q,tolerance,max_newton=min(64,max_calls-used),max_nfev=0)
        used+=len(result['jacobian_ranks'])
        certificate=dict(normal_row=k,initial_impulse=q.tolist(),
                         starting_velocity_change_m_s=float(np.max(abs(system.A@(q-p)))),
                         starting_normal_min=float(min(q[n] for n,*_ in system.contacts)),
                         starting_cone_margin_min=float(min(fr*q[n]-np.linalg.norm(q[list(tt)]) for n,tt,fr,*_ in system.contacts)),
                         final_gate=system.gate(result['impulse'],tolerance),
                         newton_residual_path=result['newton_residual_path'],jacobian_ranks=result['jacobian_ranks'])
        attempts.append(certificate)
        if result['accepted']:
            result['method']='semismooth-svd+opposing-slip-face-restart'
            return dict(result=result,attempts=attempts,newton_calls=used,warm_result=warm)
    return dict(result=warm,attempts=attempts,newton_calls=used,warm_result=warm)


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,list):return [serial(v) for v in value]
    return value


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('dump');parser.add_argument('--output');parser.add_argument('--max-nodes',type=int,default=80);parser.add_argument('--mode',choices=['gauge','opposing','baseline'],default='opposing')
    args=parser.parse_args();data=json.loads(Path(args.dump).read_text());S=System.from_dump(data)
    t=time.perf_counter()
    if args.mode=='gauge':result=explore(S,data['p'],data['tolerance_m_s'],max_nodes=args.max_nodes)
    elif args.mode=='opposing':result=opposing_slip_restart(S,data['p'],data['tolerance_m_s'])
    else:
        r=recover(S,data['p'],data['tolerance_m_s']);result=dict(result=r,attempts=r['recovery_attempts'])
    result['elapsed_s']=time.perf_counter()-t;result['analysis']=analyze(S,data['p'],data['tolerance_m_s'])
    if args.output:Path(args.output).write_text(json.dumps(serial(result),indent=2)+'\n')
    r=result['result'];print(json.dumps(dict(accepted=bool(r and r['accepted']),residual_m_s=r['residual_m_s'] if r else None,
                                           attempts=len(result['attempts']),elapsed_s=result['elapsed_s'])))


if __name__=='__main__':main()
