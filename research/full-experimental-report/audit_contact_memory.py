"""Direct body-energy, analytic oscillator, singular/scaling checks and timing."""
import argparse,hashlib,json,math,platform,time
from pathlib import Path
import numpy as np
from contact_memory import ElasticMemory


def skew(r):
    x,y,z=r;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=False)
    rng=np.random.default_rng(20261007);worst_energy=0.;worst_scale=0.;singular=0
    for dim in (2,3):
        for trial in range(100):
            m=rng.uniform(.1,3);ell=10**rng.uniform(-2,0);r=rng.uniform(-1,1,size=3)
            if dim==2:
                r[2]=0;B=np.array([[1,0,-r[1]/ell],[0,1,r[0]/ell],[0,0,1.]])
                mobility=np.diag([1/m,1/m,ell**2/rng.uniform(.1,3)])
                D=np.array([[1.,.3,0],[0,1,0],[0,0,1.]])
            else:
                x=rng.normal(size=(3,3));I=x.T@x+.2*np.eye(3)
                B=np.block([[np.eye(3),-skew(r)/ell],[np.zeros((3,3)),np.eye(3)]])
                mobility=np.block([[np.eye(3)/m,np.zeros((3,3))],[np.zeros((3,3)),ell**2*np.linalg.inv(I)]])
                n=rng.normal(size=3);n/=np.linalg.norm(n);t=rng.normal(size=3);t/=np.linalg.norm(t);s=rng.normal(size=3);s/=np.linalg.norm(s)
                if trial%5==0:s=n;singular+=1
                D=np.zeros((6,4));D[:3,0]=n;D[:3,1]=t;D[3:,2]=s;D[3:,3]=n
            W=D.T@B@mobility@B.T@D;v=rng.normal(size=B.shape[0]);u=D.T@B@v;eta=rng.normal(size=D.shape[1])
            x=rng.normal(size=W.shape);K=x.T@x+.1*np.eye(len(W))
            x=rng.normal(size=W.shape);C=x.T@x
            h=10**rng.uniform(-4,0);core=ElasticMemory(W,K,C,h)
            out,history,impulse,loss=core.step(u,eta)
            body_out=v+mobility@B.T@D@impulse
            before=.5*v@np.linalg.solve(mobility,v)+.5*eta@K@eta
            after=.5*body_out@np.linalg.solve(mobility,body_out)+.5*history@K@history
            error=abs(after+loss-before)/before;worst_energy=max(worst_energy,float(error));assert error<1e-10
            assert loss>=0 and after<=before*(1+1e-12)
            assert np.allclose(out,D.T@B@body_out)
            # Representation-only length scaling of angular contact coordinates.
            S=np.eye(len(W));S[2:,2:]*=100;invS=np.linalg.inv(S)
            scaled=ElasticMemory(S@W@S.T,invS.T@K@invS,invS.T@C@invS,h)
            so,sh,si,sl=scaled.step(S@u,S@eta)
            scale_error=max(np.linalg.norm(invS@so-out)/(1+np.linalg.norm(out)),np.linalg.norm(S.T@si-impulse)/(1+np.linalg.norm(impulse)),abs(sl-loss)/(1+loss))
            worst_scale=max(worst_scale,float(scale_error));assert scale_error<1e-9
    # Oscillator endpoint versus an independently known exact solution: actual
    # convergence checks, not a test that repeats the update implementation.
    errors=[]
    for steps in (40,80,160):
        h=1/steps;core=ElasticMemory(np.array([[1.]]),np.array([[4.]]),np.array([[0.]]),h)
        u,eta=np.array([1.]),np.array([0.])
        for _ in range(steps):u,eta,_,_=core.step(u,eta)
        err=math.hypot(u[0]-math.cos(2),eta[0]-math.sin(2)/2);errors.append(err)
    assert all(a/b>3.9 for a,b in zip(errors,errors[1:]))
    core=ElasticMemory(np.eye(4),np.eye(4)*100,np.eye(4)*.2,.001);u=np.ones(4);eta=np.zeros(4)
    timings=[]
    for _ in range(5):
        start=time.perf_counter()
        for _ in range(2000):u,eta,_,_=core.step(u,eta)
        timings.append((time.perf_counter()-start)/2000)
    assert np.all(np.isfinite(np.r_[u,eta]))
    source=Path(__file__).with_name('contact_memory.py')
    result=dict(pass_=True,planar_controls=100,spatial_controls=100,singular_direction_controls=singular,
        maximum_relative_total_energy_error=worst_energy,maximum_relative_length_scale_error=worst_scale,
        oscillator_refinement_errors=errors,second_order_convergence_pass=True,
        median_python_cached_4channel_step_microseconds=float(np.median(timings))*1e6,
        timing_scope='Local linear elastic branch with cached factor, Python/NumPy; not engine end-to-end benchmark or speedup claim',
        host=dict(machine=platform.machine(),python=platform.python_version()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        empirical_accuracy_demonstrated=False,friction_and_unilateral_branches_implemented=False,
        production_adopted=False)
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');(args.output/'contact_memory.py').write_bytes(source.read_bytes());print(json.dumps(result,indent=2))


if __name__=='__main__':main()
