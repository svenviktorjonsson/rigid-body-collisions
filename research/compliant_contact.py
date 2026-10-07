"""Passive local finite-contact reference with frozen geometry.

Three planar channels: normal speed, tangential relative speed, relative spin.
Not a full rotating-polygon collision detector or validated material model.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def elastic_slider(rate,state,stiffness,damping,normal,static,dynamic):
    """Kelvin-Voigt elastic element in series with a Coulomb-type slider.

    Static and dynamic values bound force (or moment) per normal force.
    The friction force opposes plastic slip, not elastic deformation rate.
    """
    trial=-stiffness*state-damping*rate
    if abs(trial)<=static*normal:
        force=trial;elastic_rate=rate;plastic=0.
    else:
        force=dynamic*normal*np.sign(trial)
        elastic_rate=(-force-stiffness*state)/damping
        plastic=rate-elastic_rate
    dissipation=damping*elastic_rate**2-force*plastic
    return force,elastic_rate,dissipation,plastic


def integrate_contact(K,initial,kn,cn,kt,ct,kr,cr,mu_static,mu_dynamic,
                      kappa_static,kappa_dynamic,max_time=1.,rtol=1e-10):
    K=np.asarray(K,dtype=float);initial=np.asarray(initial,dtype=float)
    if K.shape!=(3,3) or not np.all(np.isfinite(K)) or not np.allclose(K,K.T):
        raise ValueError('Require a finite symmetric 3 by 3 mobility')
    K=.5*(K+K.T)  # Remove accepted floating-point antisymmetry before energy use.
    try:np.linalg.cholesky(K)
    except np.linalg.LinAlgError as error:raise ValueError('Mobility must be positive definite') from error
    if initial.shape!=(3,) or not np.all(np.isfinite(initial)) or initial[0]>=0:
        raise ValueError('Finite approaching three-channel initial velocity required')
    if min(kn,kt,kr,ct,cr)<=0 or cn<0:raise ValueError('Positive stiffness/slider damping required')
    if not 0<=mu_dynamic<=mu_static or not 0<=kappa_dynamic<=kappa_static:
        raise ValueError('Require dynamic capacity <= static capacity')
    metric=np.linalg.inv(K)
    channels=[(kt,ct,mu_static,mu_dynamic),(kr,cr,kappa_static,kappa_dynamic)]
    # None is the continuous equal-coefficient baseline; 0 is elastic stick;
    # +/-1 is the retained direction of plastic slip until an arrest event.
    modes=[None if st==dy else 0 for _,_,st,dy in channels]
    force_tol=1e-12
    def normal_force(y):
        return max(0.,kn*max(y[3],0.)-cn*y[0]) if y[3]>=0 else 0.
    def channel_values(y,index,normal):
        stiffness,damping,static,dynamic=channels[index]
        rate=y[index+1];state=y[index+4];mode=modes[index]
        if mode is None:
            return elastic_slider(rate,state,stiffness,damping,normal,static,dynamic)
        if mode==0:
            force=-stiffness*state-damping*rate
            return force,rate,damping*rate**2,0.
        force=-dynamic*normal*mode
        elastic_rate=(-force-stiffness*state)/damping
        plastic=rate-elastic_rate
        return force,elastic_rate,damping*elastic_rate**2-force*plastic,plastic
    def rhs(t,y):
        normal=normal_force(y)
        Ft,zrate,Dt,_=channel_values(y,0,normal)
        torque,arate,Dr,_=channel_values(y,1,normal)
        compression_rate=-y[0]
        Dn=(normal-kn*y[3])*compression_rate if y[3]>=0 else 0.
        return np.r_[K@np.array([normal,Ft,torque]),compression_rate,zrate,arate,max(Dn,0)+Dt+Dr]
    def opening(t,y):return y[3]
    opening.terminal=True;opening.direction=-1
    y0=np.r_[initial,0.,0.,0.,0.]
    for index,(stiffness,damping,static,dynamic) in enumerate(channels):
        trial=-stiffness*y0[index+4]-damping*y0[index+1]
        if modes[index]==0 and abs(trial)>static*normal_force(y0)+force_tol:
            modes[index]=-np.sign(trial)
    time=0.;time_parts=[];state_parts=[];history=[];opened=False
    for phase in range(1000):
        events=[opening];labels=[('open',None)]
        for index,(stiffness,damping,static,dynamic) in enumerate(channels):
            if modes[index] is None:continue
            if modes[index]==0:
                def transition(t,y,index=index,stiffness=stiffness,damping=damping,static=static):
                    trial=-stiffness*y[index+4]-damping*y[index+1]
                    return static*normal_force(y)+force_tol-abs(trial)
                label=('yield',index)
            else:
                direction=modes[index]
                def transition(t,y,index=index,direction=direction):
                    plastic=channel_values(y,index,normal_force(y))[3]
                    return direction*plastic
                label=('arrest',index)
            transition.terminal=True;transition.direction=-1
            events.append(transition);labels.append(label)
        result=solve_ivp(rhs,(time,max_time),y0,events=events,rtol=rtol,atol=rtol*1e-3,
                         max_step=min(.0005,1/np.sqrt(kn*K[0,0])/40))
        if not result.success:raise RuntimeError(result.message)
        history.append({'start':time,'end':float(result.t[-1]),'modes':list(modes)})
        time_parts.append(result.t if phase==0 else result.t[1:])
        state_parts.append(result.y if phase==0 else result.y[:,1:])
        fired=[i for i,event in enumerate(result.t_events) if len(event)]
        if not fired:raise RuntimeError('Contact did not open within maximum time')
        time=float(result.t[-1]);y0=result.y[:,-1]
        kind,index=labels[fired[0]]
        if kind=='open':opened=True;break
        if kind=='yield':
            stiffness,damping,_,_=channels[index]
            trial=-stiffness*y0[index+4]-damping*y0[index+1]
            modes[index]=-np.sign(trial)
        else:modes[index]=0
    if not opened:raise RuntimeError('Exceeded contact transition limit; refine or regularize')
    tt=np.concatenate(time_parts);y=np.concatenate(state_parts,axis=1)
    kinetic=.5*np.einsum('it,ij,jt->t',y[:3],metric,y[:3])
    stored=.5*kn*y[3]**2+.5*kt*y[4]**2+.5*kr*y[5]**2
    initial_energy=float(kinetic[0])
    residual=kinetic+stored+y[6]-initial_energy
    return {'time':tt,'states':y,'kinetic':kinetic,'stored':stored,'phase_history':history,
            'dissipation':y[6],'energy_accounting_residual':residual,
            'post_velocity':y[:3,-1],
            'effective_normal_restitution':float(-y[0,-1]/initial[0]),
            'residual_elastic_energy_at_opening':float(stored[-1])}


def calibrate_normal_damping(e,kn,normal_mobility=1.):
    """Fit an isolated clipped Kelvin-Voigt contact, rather than treating e as
    universal under coupled oblique impact. Require 0 < e < 1.
    """
    if not 0<e<1:raise ValueError('Use positive restitution below one for finite calibration')
    def measured(c):
        K=np.diag([normal_mobility,1.,1.])
        result=integrate_contact(K,[-1,0,0],kn,c,1,1,1,1,0,0,0,0,rtol=1e-10)
        return result['effective_normal_restitution']
    lower=0.;upper=2*np.sqrt(kn/normal_mobility)
    while measured(upper)>e:upper*=2
    return brentq(lambda c:measured(c)-e,lower,upper,xtol=1e-7)
