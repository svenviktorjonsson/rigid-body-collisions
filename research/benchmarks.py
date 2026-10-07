"""Reproducible diagnostics; not a validation of 2D arbitrary-body impacts."""
from pathlib import Path
import csv
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent

def rod(elements, sigma, samples=5001):
    """Layered unit rod, lumped masses and linear springs, velocity Verlet.

    rho=A=L=1; E=1 on left half and E=4 on right half. Ends free,
    prescribed Gaussian force of unit integrated impulse at the left end.
    """
    dx=1/elements
    centers=(np.arange(elements)+.5)*dx
    stiffness=np.where(centers<.5,1.,4.)/dx
    mass=np.full(elements+1,dx);mass[[0,-1]]*=.5
    start=6*sigma;end=start+2.5
    step=min(.25*dx/2,sigma/30)
    steps=int(np.ceil(end/step));step=end/steps
    u=np.zeros(elements+1);v=np.zeros_like(u)
    def load(t):
        return np.exp(-.5*((t-start)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
    def acceleration(position,t):
        internal=stiffness*np.diff(position)
        force=np.zeros_like(position)
        force[:-1]+=internal;force[1:]-=internal
        force[0]+=load(t)
        return force/mass
    acc=acceleration(u,0)
    trace=np.empty(steps+1);trace[0]=0
    tt=np.linspace(0,end,steps+1)
    work=0.
    tic=time.perf_counter()
    for k in range(steps):
        new_u=u+step*v+.5*step*step*acc
        new_acc=acceleration(new_u,tt[k+1])
        new_v=v+.5*step*(acc+new_acc)
        work+=.5*(load(tt[k])+load(tt[k+1]))*(new_u[0]-u[0])
        u,v,acc=new_u,new_v,new_acc
        trace[k+1]=v[-1]
    seconds=time.perf_counter()-tic
    energy=.5*np.sum(mass*v*v)+.5*np.sum(stiffness*np.diff(u)**2)
    target_time=np.linspace(0,end,samples)
    return {'time':target_time,'right_velocity':np.interp(target_time,tt,trace),
            'energy':float(energy),'work':float(work),'steps':steps,
            'updates':steps*(elements+1),'seconds':seconds}


def contact_diagnostics():
    # A contact graph for a simultaneous equal-mass three-body collision.
    G=np.array([[-1.,1.,0.],[0.,-1.,1.]])
    K=G@G.T;before=np.array([1.,0.,-1.])
    impulses=np.linalg.solve(K,-2*G@before)
    after=before+G.T@impulses
    assert np.allclose(after,[-1,0,1])
    assert np.isclose(before@before,after@after)
    # Incompatible patch restitution on one body: left/center/right normals.
    J=np.array([[1.,-1.],[1.,0.],[1.,1.]])
    target=np.array([0.,1.,0.])
    best=np.linalg.lstsq(J,target,rcond=None)[0]
    incompatibility=float(np.linalg.norm(J@best-target))
    assert incompatibility>.8
    # Coupled elastic normal + tangential stopping creates kinetic energy.
    inverse=np.array([[1.,.9],[.9,1.]])
    response=np.linalg.inv(inverse);velocity=np.array([-1.,1.])
    impulse=inverse@np.array([2.,-1.])
    energy=float(velocity@impulse+.5*impulse@response@impulse)
    assert np.isclose(energy,.4)
    return {'simultaneous_chain':{'impulses':impulses.tolist(),'before':before.tolist(),'after':after.tolist()},
            'heterogeneous_restitution':{'target':target.tolist(),'least_squares_target':(J@best).tolist(),'residual_norm':incompatibility},
            'energy_counterexample':{'impulse':impulse.tolist(),'kinetic_energy_gain':energy}}


def main():
    records=[];cases=[]
    for sigma in (.015,.06,.25):
        fine=rod(2048,sigma);check=rod(1024,sigma)
        norm=np.linalg.norm(fine['right_velocity'])
        convergence=np.linalg.norm(check['right_velocity']-fine['right_velocity'])/norm
        series=[]
        for n in (8,16,32,64,128):
            coarse=rod(n,sigma)
            error=np.linalg.norm(coarse['right_velocity']-fine['right_velocity'])/norm
            energy_error=abs(coarse['energy']-fine['energy'])/fine['energy']
            row={'sigma':sigma,'elements':n,'right_velocity_relative_L2_error':float(error),
                 'final_energy_relative_error':float(energy_error),
                 'node_update_reduction':fine['updates']/coarse['updates'],
                 'runtime_ratio_observed':fine['seconds']/coarse['seconds'],
                 'reference_1024_vs_2048_error':float(convergence),
                 'coarse_work_energy_relative_residual':abs(coarse['energy']-coarse['work'])/max(coarse['work'],1e-30)}
            records.append(row);series.append((n,coarse))
        cases.append((sigma,fine,series))
    with (ROOT/'coarse-rod-results.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    (ROOT/'diagnostics.json').write_text(json.dumps(contact_diagnostics(),indent=2))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4),dpi=160)
    for ax,(sigma,fine,series) in zip(axes,cases):
        ax.plot(fine['time'],fine['right_velocity'],color='black',lw=1.5,label='2048 elements')
        for n,coarse in series:
            if n in (8,32,128):
                ax.plot(coarse['time'],coarse['right_velocity'],lw=1,label=f'{n} elements')
        ax.set_title(f'Pulse width sigma = {sigma}');ax.set_xlabel('Time')
    axes[0].set_ylabel('Right-end velocity');axes[-1].legend(fontsize=8)
    fig.suptitle('Layered 1D rod: coarse cells preserve broad pulses and distort sharp waves',fontsize=13)
    fig.tight_layout();fig.savefig(ROOT/'rod-time-traces.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),dpi=160)
    for sigma in (.015,.06,.25):
        selected=[r for r in records if r['sigma']==sigma]
        axes[0].loglog([r['elements'] for r in selected],[r['right_velocity_relative_L2_error'] for r in selected],'o-',label=f'sigma={sigma}')
        axes[1].loglog([r['node_update_reduction'] for r in selected],[r['right_velocity_relative_L2_error'] for r in selected],'o-',label=f'sigma={sigma}')
    axes[0].set_xlabel('Coarse elements');axes[0].set_ylabel('Right-end velocity relative L2 error')
    axes[1].set_xlabel('Reference/coarse node updates');axes[1].set_ylabel('Right-end velocity relative L2 error')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(ROOT/'rod-error-cost.png');plt.close(fig)
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
