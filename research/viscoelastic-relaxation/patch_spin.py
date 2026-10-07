"""Closed-form independent spin torque from a circular Hertz pressure patch.

Pure axial spin / full local sliding, constant normal load and patch radius.
No shear-history/torsional spring or mixed translation/rolling approximation.
The contact coefficient is the existing dynamic sliding friction, not a new fit.
"""
import json,math
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss


def advance(inertia,normal_load,patch_radius,mu_d,omega,dt):
    values=np.array([inertia,normal_load,patch_radius,dt],float)
    if not np.all(np.isfinite(values)) or np.any(values<=0):
        raise ValueError('positive finite inertia/load/patch radius/timestep required')
    if not np.all(np.isfinite([mu_d,omega])) or mu_d<0:
        raise ValueError('nonnegative finite dynamic friction and finite spin required')
    torque=3*math.pi/16*mu_d*normal_load*patch_radius
    scalar_delta_L_n=-np.sign(omega)*min(torque*dt,inertia*abs(omega))
    after=omega+scalar_delta_L_n/inertia
    loss=-omega*scalar_delta_L_n-.5*scalar_delta_L_n**2/inertia
    return dict(omega_after=float(after),delta_L_n=float(scalar_delta_L_n),
                instantaneous_sliding_torque_magnitude=torque,dissipation=float(loss),
                net_tangential_linear_impulse=0.)


def traction_quadrature(n,omega,N,a,mu):
    # Original distributed traction calculation, not the closed-form moment.
    t=np.cross(n,[1.,0.,0.])
    if np.linalg.norm(t)<.1:t=np.cross(n,[0.,1.,0.])
    t/=np.linalg.norm(t);s=np.cross(n,t)
    nodes,weights=leggauss(32)
    angles=(nodes+1)*math.pi/4;weights=weights*math.pi/4
    force=np.zeros(3);moment=np.zeros(3);power=0.
    p0=3*N/(2*math.pi*a*a)
    for theta,weight in zip(angles,weights):
        radial=a*math.sin(theta);dr=a*math.cos(theta)
        pressure=p0*math.cos(theta)
        for azimuth in (np.arange(32)+.5)*2*math.pi/32:
            point=radial*(math.cos(azimuth)*t+math.sin(azimuth)*s)
            local_velocity=np.cross(omega*n,point)
            traction=-mu*pressure*local_velocity/np.linalg.norm(local_velocity)
            area=radial*dr*weight*2*math.pi/32
            contribution=traction*area
            force+=contribution;moment+=np.cross(point,contribution)
            power+=float(local_velocity@contribution)
    return force,moment,power


def audit(seed=71026):
    rng=np.random.default_rng(seed);worst=0.;energy_error=0.;zero_spin_example=None
    for case in range(24):
        n=rng.normal(size=3);n/=np.linalg.norm(n)
        N=10**rng.uniform(-2,3);a=10**rng.uniform(-5,-2)
        mu=rng.uniform(.01,1);omega=rng.choice([-1,1])*rng.uniform(.1,100)
        force,moment,power=traction_quadrature(n,omega,N,a,mu)
        target=-np.sign(omega)*3*math.pi/16*mu*N*a*n
        err=max(np.linalg.norm(force)/(mu*N),np.linalg.norm(moment-target)/(mu*N*a),
                abs(power-target@(omega*n))/(mu*N*a*abs(omega)))
        worst=max(worst,float(err));assert err<1e-11
        I=10**rng.uniform(-7,-1);dt=10**rng.uniform(-6,0)
        result=advance(I,N,a,mu,omega,dt)
        before=.5*I*omega**2;after=.5*I*result['omega_after']**2
        err=abs(after+result['dissipation']-before)/before
        energy_error=max(energy_error,float(err));assert err<1e-12
        assert result['dissipation']>=0 and omega*result['omega_after']>=-1e-12
        if zero_spin_example is None:
            zero_spin_example=dict(contact_center_tangential_velocity=[0.,0.,0.],
                angular_velocity_axis=n.tolist(),spin_before=omega,spin_after=result['omega_after'],
                net_friction_force_N=force.tolist(),independent_torque_Nm=moment.tolist())
    zero=advance(.01,10.,.002,.2,0.,.1)
    assert zero['delta_L_n']==zero['omega_after']==zero['dissipation']==0
    arrested=advance(.001,10.,.002,.2,1.,10.)
    assert arrested['omega_after']==0 and arrested['dissipation']>0
    frictionless=advance(.001,10.,.002,0.,1.,10.)
    assert frictionless['omega_after']==1 and frictionless['dissipation']==0
    return dict(pass_=True,spatial_pressure_quadrature_controls=24,
        maximum_scaled_net_force_moment_work_error=worst,
        maximum_relative_body_energy_error=energy_error,
        zero_spin_no_spurious_torque=True,spin_arrest_without_reversal=True,zero_friction_preserves_spin=True,
        synthetic_center_zero_slip_example=zero_spin_example,
        new_fitted_friction_coefficients=0,full_mixed_contact_model=False,experimental_validation=False,
        physical_scope='Circular Hertz pressure, pure axial spin/full local sliding; ideal zero-external-torque arrest; actual material patch radius must be characterized')


if __name__=='__main__':print(json.dumps(audit(),indent=2))
