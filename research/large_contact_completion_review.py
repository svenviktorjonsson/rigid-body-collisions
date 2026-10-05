"""Bounded exploratory original-law recovery for large frozen contact systems.

Numerical continuation modifies trial search problems only. Every final
candidate is checked against the captured original circular law and passivity.
The public engine and frozen full-study sources are never modified here.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,minimize
from research.coulomb_trust_diagnostic import equations


def original_gate(data,p):
    A=np.asarray(data['A']);b=np.asarray(data['b']);dep=np.asarray(data['dependencies']);hi=np.asarray(data['hi'])
    f,residual=equations(data);error=float(residual(p));w=A@p-b;change=float(.5*p@A@p-b@p);scale=float(1+sum(abs(p*b)))
    normals=np.flatnonzero(dep<0)
    return dict(original_equations_accepted=bool(np.isfinite(p).all() and np.isfinite(w).all() and error<=data['tolerance_m_s'] and
                change<=data['tolerance_m_s']*scale and np.all(p[normals]>=-1e-12) and np.all(p[normals]<=hi[normals])),
                residual_m_s=error,passive_change_bound_J=change,passivity_scale=scale,normal_min=float(np.min(p[normals])))


def run(path,stages,representation,method,start,max_nfev,subset=False):
    capture=json.loads(path.read_text());A=np.asarray(capture['A']);b=np.asarray(capture['b']);dep=np.asarray(capture['dependencies']);hi=np.asarray(capture['hi']);normals=np.flatnonzero(dep<0)
    active=normals if not subset else normals[np.asarray(capture['p'])[normals]>1e-9]
    selected=np.array(sorted([int(i) for k in active for i in (k,*np.flatnonzero(dep==k))]))
    mapping={old:new for new,old in enumerate(selected)}
    data=dict(capture,A=A[np.ix_(selected,selected)].tolist(),b=b[selected].tolist(),lo=np.asarray(capture['lo'])[selected].tolist(),hi=hi[selected].tolist(),
              dependencies=[-1 if dep[i]<0 else mapping[dep[i]] for i in selected])
    target_hi=np.asarray(data['hi']);local_dep=np.asarray(data['dependencies']);local_A=np.asarray(data['A']);local_b=np.asarray(data['b']);ks=np.flatnonzero(local_dep<0);rn=1/local_A[ks,ks]
    p=np.asarray(capture['p'])[selected].copy() if start=='warm' else np.zeros(len(selected));history=[];started=time.perf_counter()
    if start=='normal-qp':
        N=local_A[np.ix_(ks,ks)];nb=local_b[ks]
        result=minimize(lambda x:.5*x@N@x-nb@x,np.zeros(len(ks)),jac=lambda x:N@x-nb,bounds=[(0,None)]*len(ks),
                        method='SLSQP',options={'ftol':1e-15,'maxiter':2000})
        p[ks]=result.x;history.append(dict(stage='normal-qp-guide',success=bool(result.success),iterations=int(result.nit)))
    for alpha in np.linspace(0,1,stages+1) if stages else [1.]:
        trial=dict(data);bounds=target_hi.copy();bounds[local_dep>=0]*=alpha;trial['hi']=bounds.tolist();base,error=equations(trial)
        def calc(p,jac=False):
            result=base(p,jac)
            if representation=='fb':
                u=p[ks]/rn;w=local_A[ks]@p-local_b[ks];length=np.hypot(u,w)
                if jac:
                    ca=1-np.divide(u,length,out=np.zeros_like(u),where=length>0);cb=1-np.divide(w,length,out=np.zeros_like(w),where=length>0)
                    result[ks]=cb[:,None]*local_A[ks];result[ks,ks]+=ca/rn
                else:result[ks]=u+w-length
            return result
        optimized=least_squares(calc,p,jac=lambda x:calc(x,True),method=method,max_nfev=max_nfev,ftol=1e-14,xtol=1e-14,gtol=1e-14)
        p=optimized.x;row=dict(alpha=float(alpha),residual_m_s=float(error(p)),nfev=int(optimized.nfev),optimality=float(optimized.optimality),optimizer_success=bool(optimized.success))
        history.append(row);print(json.dumps(row),flush=True)
    full=np.zeros(len(b));full[selected]=p;gate=original_gate(capture,full)
    result=dict(schema='large-contact-completion-trial-v1',capture=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                stages=stages,representation=representation,method=method,start=start,max_nfev_per_stage=max_nfev,subset=subset,selected_rows=selected.tolist(),
                elapsed_s=time.perf_counter()-started,history=history,p=full.tolist(),w=(A@full-b).tolist(),**gate)
    out=Path('research/completion-large-contact-review');out.mkdir(exist_ok=True)
    name=path.stem+'-'+representation+'-'+method+'-'+start+'-'+str(stages)+('-subset' if subset else '-full')+'.json'
    (out/name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(result=name,**gate,elapsed_s=result['elapsed_s'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('capture',type=Path);parser.add_argument('--stages',type=int,default=10)
    parser.add_argument('--representation',choices=['fb','natural'],default='fb');parser.add_argument('--method',choices=['trf','lm'],default='trf')
    parser.add_argument('--start',choices=['cold','warm','normal-qp'],default='normal-qp');parser.add_argument('--max-nfev',type=int,default=250);parser.add_argument('--subset',action='store_true')
    a=parser.parse_args();run(a.capture,a.stages,a.representation,a.method,a.start,a.max_nfev,a.subset)
