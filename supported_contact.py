"""Exact sustained sphere/disk contact with distinct static/dynamic friction.

Rigid body, one plane, constant normal load and drive, scalar central inertia.
Translation and rolling lie in one plane; axial spin is an independent channel.
Each branch integrates to its next slip/rolling arrest, not through it. No impact
restitution or elastic memory is substituted by this supported-contact law.
"""
from dataclasses import dataclass
import itertools
import math
import numpy as np


@dataclass(frozen=True)
class Resistance:
    mu_s: float
    mu_d: float
    mu_r: float
    rolling_length_m: float
    mu_n: float = 0.
    spin_length_m: float = 0.

    def __post_init__(self):
        values=[self.mu_s,self.mu_d,self.mu_r,self.rolling_length_m,self.mu_n,self.spin_length_m]
        if not all(math.isfinite(x) and x>=0 for x in values) or self.mu_d>self.mu_s:
            raise ValueError('Finite nonnegative parameters and mu_d <= mu_s required')
        if (self.mu_r and not self.rolling_length_m) or (self.mu_n and not self.spin_length_m):
            raise ValueError('Nonzero angular resistance requires its physical moment length')


def _branch(v,w,m,I,R,N,drive,law,tolerance):
    """Find static/onset/continued-motion constraints without arbitrary directions."""
    u=v-R*w;A=1/m+R*R/I;cap=law.mu_r*law.rolling_length_m*N
    signs=lambda x,tol: [int(np.sign(x))] if abs(x)>tol else [0,-1,1]
    force_tolerance=64*np.finfo(float).eps*max(N,abs(drive),cap/R,1e-300)
    for su,sw in itertools.product(signs(u,tolerance),signs(w,tolerance/R)):
        if su==sw==0:f=-drive;M=R*f
        elif su==0:
            M=-cap*sw;f=(R*M/I-drive/m)/A
        elif sw==0:f=-law.mu_d*N*su;M=R*f
        else:f=-law.mu_d*N*su;M=-cap*sw
        if su==0 and abs(f)>law.mu_s*N+force_tolerance:continue
        if sw==0 and abs(M)>cap+R*force_tolerance:continue
        dv=(drive+f)/m;dw=(-R*f+M)/I;du=dv-R*dw
        acceleration_tolerance=force_tolerance*max(A,R/I)
        if abs(u)<=tolerance and su and su*du<=acceleration_tolerance:continue
        if abs(w)<=tolerance/R and sw and sw*dw<=acceleration_tolerance/R:continue
        return f,M,dv,dw,su,sw
    raise RuntimeError('No admissible supported-contact branch; no fallback law')


def advance_planar(*,mass_kg,inertia_kg_m2,radius_m,normal_load_N,drive_force_N,
                   velocity_m_s,omega_rad_s,spin_rad_s=0.,duration_s,material,
                   velocity_tolerance=1e-12,max_events=32):
    """Relative translation, rolling spin and axial spin on a stationary plane.

    The rotational inertia is an explicit input: disks, solid spheres and shells
    have different values. mu_r*a_r and mu_n*a_n are physical moment lengths;
    ell never appears in this calculation. Return impulses separately from body
    changes, plus exact dissipated work and all branch transitions.
    """
    m,I,R,N,F=mass_kg,inertia_kg_m2,radius_m,normal_load_N,drive_force_N
    values=[m,I,R,N,F,velocity_m_s,omega_rad_s,spin_rad_s,duration_s,velocity_tolerance]
    if not all(math.isfinite(x) for x in values) or min(m,I,R,duration_s,velocity_tolerance)<=0 or N<0:
        raise ValueError('Finite inputs, positive mass/inertia/radius/duration/tolerance and nonnegative normal load required')
    if not isinstance(material,Resistance) or type(max_events)!=int or max_events<1:raise ValueError('Resistance and positive integer event budget required')
    v=float(velocity_m_s);w=float(omega_rad_s);v0=v;w0=w
    t=0.;distance=0.;p=0.;L=0.;sliding_loss=0.;rolling_loss=0.;events=[]
    while t<duration_s:
        if len(events)>=max_events:raise RuntimeError('Supported contact exceeded branch budget; no accepted result')
        u=v-R*w
        if abs(u)<=velocity_tolerance:v=R*w;u=0.
        if abs(w)<=velocity_tolerance/R:w=0.
        f,M,dv,dw,su,sw=_branch(v,w,m,I,R,N,F,material,velocity_tolerance)
        remaining=duration_s-t;h=remaining;stop_u=False;stop_w=False
        du=dv-R*dw
        candidates=[]
        if su and u*du<0:candidates.append((-u/du,'slip_arrest'))
        if sw and w*dw<0:candidates.append((-w/dw,'rolling_arrest'))
        for interval,label in candidates:
            if interval>0 and interval<h:h=interval
        for interval,label in candidates:
            if abs(interval-h)<=16*np.finfo(float).eps*max(duration_s,h):
                if label=='slip_arrest':stop_u=True
                else:stop_w=True
        if h<=0:raise RuntimeError('Nonadvancing supported-contact event')
        dx=v*h+.5*dv*h*h;angle=w*h+.5*dw*h*h
        slip_integral=u*h+.5*du*h*h
        sliding_loss-=f*slip_integral;rolling_loss-=M*angle
        distance+=dx;p+=f*h;L+=M*h
        events.append(dict(start_s=t,duration_s=h,start_velocity_m_s=v,start_omega_rad_s=w,
                           translation_acceleration_m_s2=dv,rolling_acceleration_rad_s2=dw,
                           force_N=f,independent_rolling_moment_Nm=M,
                           slip_sign=su,rolling_sign=sw,slip_arrest=stop_u,rolling_arrest=stop_w))
        v+=dv*h;w+=dw*h;t+=h
        if stop_w:w=0.
        if stop_u:v=R*w
        if duration_s-t<=8*np.finfo(float).eps*duration_s:t=duration_s
    spin0=float(spin_rad_s);spin_cap=material.mu_n*material.spin_length_m*N*duration_s
    spin_impulse=-np.sign(spin0)*min(spin_cap,I*abs(spin0));spin=spin0+spin_impulse/I
    spin_loss=.5*I*(spin0*spin0-spin*spin)
    initial=.5*m*v0*v0+.5*I*(w0*w0+spin0*spin0)
    final=.5*m*v*v+.5*I*(w*w+spin*spin)
    external_work=F*distance;loss=sliding_loss+rolling_loss+spin_loss
    residual=final-initial-external_work+loss
    scale=max(initial,final,abs(external_work),loss,1e-300)
    if min(sliding_loss,rolling_loss,spin_loss)<-1e-11*scale or abs(residual)>2e-10*scale:
        raise RuntimeError('Supported-contact energy gate rejected result')
    return dict(velocity_m_s=v,omega_rad_s=w,spin_rad_s=spin,distance_m=distance,
                tangent_impulse_Ns=p,normal_impulse_Ns=N*duration_s,
                independent_rolling_impulse_Nms=L,independent_spin_impulse_Nms=float(spin_impulse),
                body_angular_change_Nms=I*(w-w0),body_spin_change_Nms=I*(spin-spin0),
                initial_kinetic_J=initial,final_kinetic_J=final,external_work_J=external_work,
                sliding_loss_J=sliding_loss,rolling_loss_J=rolling_loss,spin_loss_J=float(spin_loss),
                energy_residual_J=residual,events=events)


def advance_spatial(*,normal,direction,velocity,omega,plane_velocity=(0.,0.,0.),**options):
    """Rotated spatial embedding; reject noncollinear slip/rolling or normal impact.

    Direction is a declared motion plane, not a replacement definition of t/s.
    Full instantaneous t=u/|u| and s=omega/|omega| are evaluated in the returned
    trajectory's branch states. At zero motion the constraints supply reactions.
    Angular impulse includes r cross delta p PLUS the independent delta L.
    """
    n,d,v,w,U=[np.asarray(x,float) for x in [normal,direction,velocity,omega,plane_velocity]]
    if any(x.shape!=(3,) or not np.all(np.isfinite(x)) for x in [n,d,v,w,U]):raise ValueError('Finite three-vectors required')
    if not np.isclose(n@n,1,atol=1e-12,rtol=0) or not np.isclose(d@d,1,atol=1e-12,rtol=0) or abs(n@d)>1e-12:
        raise ValueError('Unit orthogonal plane normal and motion direction required')
    axis=np.cross(n,d);relative=v-U;wr=float(w@axis);wn=float(w@n);vr=float(relative@d)
    scale=max(np.linalg.norm(relative),options['radius_m']*np.linalg.norm(w),1.)
    if np.linalg.norm(relative-vr*d)>1e-11*scale or np.linalg.norm(w-wr*axis-wn*n)>1e-11*scale/options['radius_m'] or abs(U@n)>1e-12:
        raise ValueError('This exact branch requires planar translation/rolling, no normal impact, and translating fixed-orientation support')
    result=advance_planar(velocity_m_s=vr,omega_rad_s=wr,spin_rad_s=wn,**options)
    R=options['radius_m'];m=options['mass_kg'];I=options['inertia_kg_m2'];arm=-R*n
    delta_p=result['normal_impulse_Ns']*n+result['tangent_impulse_Ns']*d
    delta_L=result['independent_rolling_impulse_Nms']*axis+result['independent_spin_impulse_Nms']*n
    final_v=U+result['velocity_m_s']*d;final_w=result['omega_rad_s']*axis+result['spin_rad_s']*n
    angular_change=np.cross(arm,delta_p)+delta_L
    if np.linalg.norm(angular_change-I*(final_w-w))>1e-10*max(np.linalg.norm(angular_change),1e-300):
        raise RuntimeError('Independent angular impulse plus lever moment gate failed')
    support_work=float(U@delta_p)
    spin_torque=options['material'].mu_n*options['material'].spin_length_m*options['normal_load_N']
    spin_rate=-np.sign(wn)*spin_torque/I
    spin_stop=abs(wn/spin_rate) if spin_rate else float('inf')
    frames=[]
    for event in result['events']:
        times=[event['start_s']]
        if event['start_s']<spin_stop<event['start_s']+event['duration_s']:times.append(spin_stop)
        for time in times:
            h=time-event['start_s'];local_v=event['start_velocity_m_s']+h*event['translation_acceleration_m_s2']
            local_w=event['start_omega_rad_s']+h*event['rolling_acceleration_rad_s2']
            axial=wn+spin_rate*time if time<spin_stop else 0.
            motion=(local_v-R*local_w)*d;rotation=local_w*axis+axial*n
            speed=np.linalg.norm(motion);angular_speed=np.linalg.norm(rotation)
            moment=event['independent_rolling_moment_Nm']*axis+(I*spin_rate if time<spin_stop else 0.)*n
            t=motion/speed if speed>options.get('velocity_tolerance',1e-12) else None
            s=rotation/angular_speed if R*angular_speed>options.get('velocity_tolerance',1e-12) else None
            moment_map=np.column_stack([s,n]) if s is not None else n[:,None]
            coefficients=np.linalg.lstsq(moment_map,moment,rcond=None)[0]
            residual=np.linalg.norm(moment-moment_map@coefficients)
            # At complete zero spin the new constraint branch supplies a reaction.
            # If full s remains defined but is parallel n, an arbitrary transverse
            # static moment is outside the user's span: refuse that spatial case.
            if s is not None and residual>1e-10*max(np.linalg.norm(moment),1e-300):
                raise ValueError('Partial angular arrest requires a static transverse couple outside the supplied s/n span; unsupported directional branch')
            frames.append(dict(time_s=time,t=None if t is None else t.tolist(),s=None if s is None else s.tolist(),
                               force_n_N=options['normal_load_N'],force_t_N=None if t is None else float(event['force_N']*d@t),
                               independent_moment=moment.tolist(),angular_span_residual_Nm=float(residual),
                               static_linear_reaction=t is None,static_angular_reaction=s is None))
    result.update(velocity=final_v.tolist(),omega=final_w.tolist(),linear_impulse=delta_p.tolist(),
                  independent_angular_impulse=delta_L.tolist(),body_angular_change=angular_change.tolist(),
                  support_work_J=support_work,
                  directional_branch_frames=frames,
                  position_change=(options['duration_s']*U+result['distance_m']*d).tolist())
    return result
