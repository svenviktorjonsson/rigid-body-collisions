"""Energy-accounted sphere/plane elastic wrench prototype (SI units).

This is a material hypothesis, not a calibrated rubber law. Tangential force
and independent normal-axis couple share one normal-load yield budget. A
compression-weighted potential vanishes at lift-off. Its normal derivative is
included in the force, so releasing contact history does not delete energy.
The optional constant-stiffness benchmark explicitly records any separation
loss; matched oscillator cases have zero residual. The shared L2 yield is a
phenomenological generalized slider, not a derived exact finite-patch surface.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Material:
    mass: float = 1.0
    radius: float = 0.1
    normal_stiffness: float = 1e5
    tangent_stiffness: float = 1e8
    twist_stiffness: float = 1e6
    effective_length: float = 0.02
    friction: float = 1.0
    normal_damping: float = 0.0
    compression_exponent: int = 2

    def __post_init__(self):
        if self.compression_exponent not in (0, 2):
            raise ValueError('compression_exponent must be 0 (linear benchmark) or 2')
        for name in ('mass', 'radius', 'normal_stiffness', 'effective_length'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('tangent_stiffness', 'twist_stiffness', 'friction', 'normal_damping'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')

    @property
    def inertia(self):
        return 0.4 * self.mass * self.radius ** 2

    @property
    def stiffness(self):
        return np.array([self.tangent_stiffness, self.tangent_stiffness,
                         self.twist_stiffness / self.effective_length ** 2])


@dataclass(frozen=True)
class Plane:
    """Fixed plane n dot x = offset; n points into the admissible half-space."""
    normal: tuple = (0., 0., 1.)
    offset: float = 0.
    name: str = 'floor'

    def __post_init__(self):
        n = np.asarray(self.normal, dtype=float)
        if n.shape != (3,) or not np.all(np.isfinite(n)) or not np.isclose(np.linalg.norm(n), 1., atol=1e-12, rtol=0.):
            raise ValueError('plane normal must be a unit three-vector')
        if not np.isfinite(self.offset):
            raise ValueError('plane offset must be finite')

    @property
    def basis(self):
        n = np.asarray(self.normal)
        seed = np.eye(3)[np.argmin(abs(n))]
        t1 = np.cross(n, seed); t1 /= np.linalg.norm(t1)
        return np.stack([t1, np.cross(n, t1)], axis=1)


def yield_state(material, plane, x, v, omega, history):
    """Yield gap and outward elastic loading drive for hybrid mode events."""
    m=material;n=np.asarray(plane.normal);delta=max(0.,m.radius-(n@x-plane.offset))
    if delta==0 or m.friction==0:return -m.friction*m.normal_stiffness*delta,0.
    f=(delta/m.radius)**m.compression_exponent;elastic=f*m.stiffness*np.asarray(history)
    magnitude=np.linalg.norm(elastic);cap=m.friction*m.normal_stiffness*delta
    if magnitude==0:return -cap,0.
    u=np.r_[plane.basis.T@(v+np.cross(omega,-m.radius*n)),m.effective_length*(n@omega)]
    delta_dot=-n@v
    raw=m.compression_exponent*delta_dot/delta*elastic+f*m.stiffness*u
    drive=float(elastic@raw/magnitude-m.friction*m.normal_stiffness*delta_dot)
    return float(magnitude-cap),drive


def contact(material, plane, x, v, omega, history, *, plastic_active=None):
    """Return force, independent couple, strain rate, energy, dissipation, load.

    Strain is [two tangent displacements, a_eff * relative twist angle].
    Plastic flow is associative in this scaled strain. Yield uses the base
    elastic normal load; the full normal force is at least this large, so
    the shared force/couple bound is conservative. Normal damping acts only
    during compression (a passive, explicitly chosen material law). One coefficient covers stick/slide.
    """
    m = material; n = np.asarray(plane.normal); T = plane.basis
    delta = max(0., m.radius - (n @ x - plane.offset))
    if delta == 0:
        return np.zeros(3), np.zeros(3), np.zeros(3), 0., 0., 0., 0.
    delta_dot = -n @ v
    f = (delta / m.radius) ** m.compression_exponent
    h = np.asarray(history); K = m.stiffness
    strain_velocity = np.r_[T.T @ (v + np.cross(omega, -m.radius*n)),
                            m.effective_length*(n @ omega)]
    # Gradients of the *same* potential provide both normal and shear forces.
    elastic = f * K * h
    history_energy = 0.5 * f * np.dot(h, K*h)
    base_normal = m.normal_stiffness*delta + m.normal_damping*max(0.,delta_dot)
    normal = base_normal + m.compression_exponent*history_energy/delta
    cap = m.friction * m.normal_stiffness*delta
    magnitude = np.linalg.norm(elastic)
    rate = strain_velocity.copy()
    plastic_power = 0.
    if m.friction == 0:
        # No material shear/twist contact; no artificial spring memory.
        elastic[:] = 0.; history_energy = 0.; normal = base_normal
        rate[:] = 0.
    elif magnitude > 0 and (plastic_active is True or
                           (plastic_active is None and magnitude >= cap*(1.-1e-9))):
        direction = elastic/magnitude
        # Differentiate |f K h| <= mu*N_base. At the boundary, remove
        # outward motion by nonnegative associated plastic flow.
        raw_rate = m.compression_exponent*delta_dot/delta*elastic + f*K*strain_velocity
        # Elastic load defines the conservative shared yield budget.
        cap_dot = m.friction*m.normal_stiffness*delta_dot
        denominator = f*np.dot(direction, K*direction)
        multiplier = max(0., (np.dot(direction, raw_rate)-cap_dot)/denominator) if denominator else 0.
        rate -= multiplier*direction
        plastic_power = magnitude*multiplier
    force = normal*n - T @ elastic[:2]
    couple = -m.effective_length*elastic[2]*n
    normal_loss = (base_normal-m.normal_stiffness*delta)*delta_dot
    stored = .5*m.normal_stiffness*delta**2 + history_energy
    # Use elastic load for the shared yield: it does not grant damping extra
    # friction capacity and avoids a force/acceleration algebraic loop.
    return force, couple, rate, stored, normal_loss+plastic_power, normal, magnitude-m.friction*m.normal_stiffness*delta


def ballistic_segment(material,planes,gravity,start,end,initial):
    """Exact free motion up to the earliest inward geometric crossing."""
    remaining=end-start;crossings=[]
    for j,p in enumerate(planes):
        n=np.asarray(p.normal)
        a=-.5*n@gravity;b=-n@initial[3:6];c=material.radius-(n@initial[:3]-p.offset)
        roots=np.roots([a,b,c]) if a!=0 else ([-c/b] if b!=0 else [])
        for root in roots:
            if np.isreal(root):
                value=float(root)
                if value>0 and value<=remaining and b+2*a*value>0:crossings.append((value,j))
    flight=min((value for value,j in crossings),default=remaining);stop=start+flight
    def solution(times):
        dt=np.asarray(times)-start
        if dt.ndim==0:
            state=initial.copy();state[:3]+=initial[3:6]*dt+.5*gravity*dt*dt
            state[3:6]+=gravity*dt;return state
        state=np.repeat(initial[:,None],len(dt),axis=1)
        state[:3]+=initial[3:6,None]*dt+.5*gravity[:,None]*dt*dt
        state[3:6]+=gravity[:,None]*dt;return state
    t_events=[np.array([]) for _ in planes]
    for value,j in crossings:
        if abs(value-flight)<=1e-13*max(1.,flight):t_events[j]=np.array([stop])
    times=np.array([start,stop])
    return SimpleNamespace(t=times,y=solution(times),sol=solution,t_events=t_events,success=True)


def simulate(material, *, position=(0.,0.,.2), velocity=(0.,0.,-1.),
             omega=(0.,0.,0.), planes=(Plane(),), gravity=(0.,0.,0.),
             duration=.1, sample_dt=.0002, max_step=.0001,
             rtol=1e-9, atol=1e-11, max_rhs_evaluations=200000):
    """Hybrid elastic/plastic DOP853 contact and exact ballistic free motion.

    Report collision linear impulse and independent couple impulse separately.
    The result also retains every solver's steps, contact events and energy
    channels. Reducing max_step/tolerance supplies a numerical reference; it
    does not establish an experimentally authenticated material.
    """
    m = material; planes = tuple(planes); nplanes=len(planes)
    gravity=np.asarray(gravity, dtype=float)
    for name, val in [('position', position), ('velocity', velocity), ('omega', omega), ('gravity', gravity)]:
        arr=np.asarray(val, dtype=float)
        if arr.shape != (3,) or not np.all(np.isfinite(arr)): raise ValueError(name+' must be a finite three-vector')
    if duration <= 0 or sample_dt <= 0 or max_step <= 0 or rtol <= 0 or atol <= 0: raise ValueError('integration controls must be positive')
    hi=9; di=hi+3*nplanes; pi=di+1; li=pi+3
    y=np.zeros(li+3); y[:9]=np.r_[position,velocity,omega]
    active=[]
    for p in planes:
        n=np.asarray(p.normal);gap=m.radius-(n@y[:3]-p.offset);approach=-n@y[3:6]
        active.append(bool(gap>0 or (gap==0 and (approach>0 or (approach==0 and -n@gravity>0)))))
    plastic_active=[False]*nplanes
    if not isinstance(max_rhs_evaluations,int) or max_rhs_evaluations<1: raise ValueError('max_rhs_evaluations must be a positive integer')
    rhs_evaluations=0
    segments=[]; events=[]; material_events=[]; internal=[]; right_limits=[]; t=0.; segments_limit=10000
    free_flight_segments=0
    def rhs(_, state):
        nonlocal rhs_evaluations
        rhs_evaluations += 1
        if rhs_evaluations>max_rhs_evaluations:
            raise RuntimeError('elastic prototype exceeded RHS evaluation budget; no accepted result')
        derivative=np.zeros_like(state); derivative[:3]=state[3:6]
        totalforce=np.zeros(3); totaltorque=np.zeros(3); couple_total=np.zeros(3)
        for j,p in enumerate(planes):
            if not active[j]: continue
            h=state[hi+3*j:hi+3*j+3]
            force,couple,rate,_,loss,_,_=contact(m,p,state[:3],state[3:6],state[6:9],h,
                                              plastic_active=plastic_active[j])
            totalforce+=force; couple_total+=couple
            totaltorque+=np.cross(-m.radius*np.asarray(p.normal),force)+couple
            derivative[hi+3*j:hi+3*j+3]=rate; derivative[di]+=loss
        derivative[3:6]=gravity+totalforce/m.mass
        derivative[6:9]=totaltorque/m.inertia
        derivative[pi:pi+3]=totalforce; derivative[li:li+3]=couple_total
        return derivative
    while t < duration-1e-14:
        event_functions=[];event_kinds=[]
        for j,p in enumerate(planes):
            def event(_,state,p=p,j=j):
                n=np.asarray(p.normal);gap=m.radius-(n@state[:3]-p.offset)
                # An inactive grazing contact has no crossing: returning zero
                # forever would cause spurious alternating t=0 transitions.
                if not active[j] and gap==0 and -n@state[3:6]<=0:return -1e-15*m.radius
                return gap
            event.terminal=True; event.direction=-1 if active[j] else 1
            event_functions.append(event);event_kinds.append(('contact',j))
            if active[j] and m.friction>0 and np.any(m.stiffness>0):
                mode='release' if plastic_active[j] else 'yield'
                def mode_event(_,state,j=j,p=p,mode=mode):
                    h=state[hi+3*j:hi+3*j+3]
                    gap,drive=yield_state(m,p,state[:3],state[3:6],state[6:9],h)
                    return drive if mode=='release' else gap
                mode_event.terminal=True;mode_event.direction=-1 if mode=='release' else 1
                event_functions.append(mode_event);event_kinds.append((mode,j))
        if not any(active):
            solution=ballistic_segment(m,planes,gravity,t,duration,y)
            free_flight_segments+=1
        else:
            solution=solve_ivp(rhs,(t,duration),y,method='DOP853',rtol=rtol,atol=atol,
                               max_step=max_step,dense_output=True,events=event_functions)
        if not solution.success: raise RuntimeError(solution.message)
        segments.append(solution); internal.extend(solution.t.tolist())
        t=float(solution.t[-1]); y=solution.y[:,-1].copy()
        triggered=[event_kinds[i] for i,e in enumerate(solution.t_events) if len(e)]
        if not triggered: break
        for kind,j in triggered:
            if kind!='contact':
                plastic_active[j]=kind=='yield'
                material_events.append(dict(time_s=t,plane=planes[j].name,kind=kind))
                continue
            entering=not active[j]; active[j]=entering
            plastic_active[j]=False
            separation_loss = 0.
            if not entering and m.compression_exponent == 0:
                h = y[hi+3*j:hi+3*j+3]
                separation_loss = .5*np.dot(h, m.stiffness*h)
                y[di] += separation_loss
            events.append({'separation_loss_J': separation_loss, 'time_s':t,'plane':planes[j].name,'kind':'entry' if entering else 'lift_off',
                           'velocity_m_s':y[3:6].tolist(),'omega_rad_s':y[6:9].tolist(),
                           'linear_impulse_N_s':y[pi:pi+3].tolist(),
                           'couple_impulse_N_m_s':y[li:li+3].tolist()})
            y[hi+3*j:hi+3*j+3]=0.
        right_limits.append((t,y.copy()))
        if len(segments)>segments_limit: raise RuntimeError('too many contact transitions')
    times=np.r_[np.arange(0.,duration,sample_dt),duration]
    states=np.empty((len(times),len(y))); index=0
    for i, ti in enumerate(times):
        while index<len(segments)-1 and ti>segments[index].t[-1]+1e-14: index+=1
        states[i]=segments[index].sol(ti)
        for event_time,event_state in right_limits:
            if abs(ti-event_time)<=1e-14:
                states[i]=event_state; break
    kinetic=.5*m.mass*np.sum(states[:,3:6]**2,axis=1)+.5*m.inertia*np.sum(states[:,6:9]**2,axis=1)
    potential=-m.mass*(states[:,:3]@gravity)
    stored=np.zeros(len(times)); peak_yield=0.
    normal=np.zeros((len(times),nplanes)); history_store=np.zeros_like(normal)
    tangent_store=np.zeros_like(normal); twist_store=np.zeros_like(normal); normal_store=np.zeros_like(normal)
    for i,state in enumerate(states):
        for j,p in enumerate(planes):
            h=state[hi+3*j:hi+3*j+3]
            _,_,_,U,_,N,yield_excess=contact(m,p,state[:3],state[3:6],state[6:9],h)
            stored[i]+=U; normal[i,j]=N
            delta=max(0.,m.radius-(np.asarray(p.normal)@state[:3]-p.offset))
            normal_store[i,j]=.5*m.normal_stiffness*delta**2
            history_store[i,j]=U-normal_store[i,j]
            f=(delta/m.radius)**m.compression_exponent if delta>0 else 0.
            tangent_store[i,j]=.5*f*np.dot(h[:2],m.stiffness[:2]*h[:2])
            twist_store[i,j]=.5*f*m.stiffness[2]*h[2]**2
            peak_yield=max(peak_yield,yield_excess)
    # Bound diagnostics include every accepted integration step as well as
    # presentation samples. This remains a numerical check, not a continuous
    # mathematical supremum or an experimental validation.
    for segment in segments:
        for state in segment.y.T:
            for j,p in enumerate(planes):
                h=state[hi+3*j:hi+3*j+3]
                excess=contact(m,p,state[:3],state[3:6],state[6:9],h)[-1]
                peak_yield=max(peak_yield,excess)
    total=kinetic+potential+stored
    accounted=total+states[:,di]
    return {'times':times,'states':states[:,:9], 'strain':states[:,hi:di],
            'internal_times':np.concatenate([segment.t for segment in segments]),
            'internal_states':np.concatenate([segment.y.T for segment in segments]),
            'kinetic_J':kinetic,'potential_J':potential,'stored_J':stored,
            'history_stored_J':history_store, 'normal_stored_J':normal_store,
            'tangent_stored_J':tangent_store,'twist_stored_J':twist_store, 'dissipated_J':states[:,di],
            'energy_residual_J':accounted-accounted[0], 'normal_force_N':normal,
            'linear_impulse_N_s':states[:,pi:pi+3], 'couple_impulse_N_m_s':states[:,li:li+3],
            'events':events,'material_events':material_events,'free_flight_segments':free_flight_segments,
            'internal_steps':len(internal),'rhs_evaluations':rhs_evaluations,
            'max_yield_excess_N':peak_yield}
