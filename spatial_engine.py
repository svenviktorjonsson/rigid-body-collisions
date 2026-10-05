"""Pinned Float64 Bullet 3D adapter. SI volume density; XYZW quaternions.

Authored position is the physical COM. Shape coordinates are recentered to their
aggregate COM; output orientation refers to authored body axes, not inertia axes.
Friction is the upstream pyramid or the project circular Coulomb lane;
coefficients use product mixing (upstream clamps pair friction to 10).
Restitution also uses product mixing. No calibrated material/rolling law claimed.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

BULLET_COMMIT = '2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5'
BINARY = Path(__file__).parent / 'build/spatial/spatial_runner'
PRESETS = {'fast': (1, 8, 'sequential'), 'accurate': (4, 64, 'coupled')}


def vector(value, n, label):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f'{label} must contain {n} finite values')
    return a


def positive(value, label, zero=False):
    x = float(value)
    if not np.isfinite(x) or x < 0 or (x == 0 and not zero):
        raise ValueError(f'Invalid {label}')
    return x


def rotation(q):
    q = vector(q, 4, 'quaternion')
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-8):
        raise ValueError('Unit XYZW quaternion required')
    return Rotation.from_quat(q)


def moments(shape):
    """Volume, centroid, inertia per unit density, conservative feature scale."""
    kind = shape['kind']
    if kind == 'sphere':
        r = positive(shape['radius'], 'radius')
        v = 4*np.pi*r**3/3
        return v, np.zeros(3), np.eye(3)*2*v*r*r/5, r
    if kind == 'box':
        h = vector(shape['half_extents'], 3, 'half extents')
        if np.min(h) <= 0:
            raise ValueError('Positive half extents required')
        v = 8*np.prod(h)
        return v, np.zeros(3), np.diag(v*(np.sum(h*h)-h*h)/3), np.min(h)
    if kind != 'hull':
        raise ValueError('Shapes must be box, sphere or convex hull')
    points = np.asarray(shape['vertices'], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Finite 3D hull vertices required')
    hull = ConvexHull(points)
    origin = points[hull.vertices].mean(axis=0)
    volume = 0.; first = np.zeros(3); second = np.zeros((3, 3))
    for face in hull.simplices:
        tet = np.vstack([origin, points[face]])
        v = abs(np.linalg.det((tet[1:]-tet[0]).T))/6
        s = tet.sum(axis=0)
        volume += v; first += v*s/4
        second += v*(np.outer(s, s)+tet.T@tet)/20
    center = first/volume
    covariance = second-volume*np.outer(center, center)
    inertia = np.trace(covariance)*np.eye(3)-covariance
    # Half the minimum support width over hull face normals.
    projections = points @ hull.equations[:, :3].T
    feature = np.min(np.ptp(projections, axis=0))/2
    return volume, center, inertia, feature


def prepare(scene):
    bodies=[]; masses=[]; tensors=[]; axes=[]; features=[]
    for authored in scene['bodies']:
        if any(key in authored for key in ('mass','inertia','inertia_tensor')):
            raise ValueError('3D mass/inertia are geometry-derived; set shape volume density')
        if any(authored.get(key,0)!=0 for key in ('rolling','rolling_friction','twisting','spinning_friction')):
            raise ValueError('Rolling/twisting moments are not implemented in this adapter')
        kind = authored.get('type', 'dynamic')
        if kind not in ('dynamic', 'static', 'kinematic'):
            raise ValueError('Invalid body type')
        parts=[]; mass=0.; first=np.zeros(3)
        for shape in authored['shapes']:
            if any(key in shape for key in ('friction','restitution','rolling','rolling_friction','twisting','spinning_friction')):
                raise ValueError('3D per-fixture material laws are not supported; set body friction/restitution')
            v, c, inertia, feature = moments(shape)
            rho = positive(shape.get('density', 1), 'volume density')
            R = rotation(shape.get('orientation', [0,0,0,1])).as_matrix()
            offset = vector(shape.get('center', [0,0,0]),3,'shape center')
            c = R@c+offset; m=v*rho
            parts.append((shape,m,c,R@inertia@R.T*rho,feature,R,offset))
            mass+=m;first+=m*c
        if not parts:
            raise ValueError('Body needs shapes')
        com=first/mass; I=np.zeros((3,3))
        for _,m,c,J,_,_,_ in parts:
            d=c-com; I+=J+m*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
        values, Q = np.linalg.eigh(I)
        if np.linalg.det(Q)<0: Q[:,0]*=-1
        if np.min(values)<=0: raise ValueError('Positive definite inertia required')
        original=rotation(authored.get('orientation',[0,0,0,1]))
        native_shapes=[];radius=0
        for s,m,c,J,feature,R,offset in parts:
            local=Q.T@(offset-com)
            native={k:v for k,v in s.items() if k not in ('density','center','orientation')}
            native.update(center=local.tolist(),orientation=Rotation.from_matrix(Q.T@R).as_quat().tolist())
            native_shapes.append(native);features.append(feature)
            if s['kind']=='sphere': tip=s['radius']
            elif s['kind']=='box': tip=np.linalg.norm(s['half_extents'])
            else: tip=np.max(np.linalg.norm(np.asarray(s['vertices']),axis=1))
            radius=max(radius,np.linalg.norm(local)+tip)
        velocity=vector(authored.get('velocity',[0,0,0]),3,'velocity')
        omega=vector(authored.get('omega',[0,0,0]),3,'omega')
        if kind=='static' and (np.any(velocity) or np.any(omega)):
            raise ValueError('Static bodies cannot move')
        schedule=authored.get('velocity_schedule',[])
        if schedule and kind!='kinematic':raise ValueError('Only kinematic schedules supported')
        commands=[]; previous=-1
        for command in schedule:
            t=positive(command['time_s'],'schedule time',zero=True)
            if not previous<t<float(scene['duration']):raise ValueError('Schedule times must increase within horizon')
            commands.append(dict(time_s=t,velocity=vector(command.get('velocity',[0,0,0]),3,'schedule velocity').tolist(),omega=vector(command.get('omega',[0,0,0]),3,'schedule omega').tolist()));previous=t
        friction=positive(authored.get('friction',.5),'friction',zero=True)
        restitution=positive(authored.get('restitution',0),'restitution',zero=True)
        if restitution>1:raise ValueError('Restitution must be in [0,1]')
        bodies.append(dict(type=kind,mass=mass if kind=='dynamic' else 0,principal_inertia=values.tolist() if kind=='dynamic' else [0,0,0],position=vector(authored.get('position',[0,0,0]),3,'position').tolist(),orientation=(original*Rotation.from_matrix(Q)).as_quat().tolist(),velocity=velocity.tolist(),omega=omega.tolist(),friction=friction,restitution=restitution,shapes=native_shapes,radius=radius,schedule=commands))
        masses.append(mass if kind=='dynamic' else 0);tensors.append(I.tolist());axes.append(Q)
    if not bodies:raise ValueError('At least one body required')
    return bodies,masses,tensors,axes,min(features)


def run(scene, *, dt=1/120, primary_steps=4, iterations=64, solver='coupled', travel_fraction=.15, kinematic_contact_phase="end", position_stabilization="split", preassembly_elimination=True, contact_tolerance_m_s=1e-8, contact_slop_m=1e-9, rejected_contact_path=None, contact_recovery=True, binary=BINARY):
    duration=positive(scene['duration'],'duration');dt=positive(dt,'dt')
    frames=round(duration/dt)
    if frames<1 or not np.isclose(frames*dt,duration,rtol=1e-10,atol=1e-12):raise ValueError('Duration must match output frames')
    for value,label in ((primary_steps,'primary_steps'),(iterations,'iterations')):
        if type(value)!=int or not 1<=value<=4096:raise ValueError(f'Invalid {label}')
    if type(contact_recovery) is not bool:raise ValueError('contact_recovery must be bool')
    if type(preassembly_elimination) is not bool:raise ValueError('preassembly_elimination must be bool')
    if solver not in ('coupled','sequential','adaptive','normal_coupled','coulomb'):raise ValueError('Invalid solver')
    travel_fraction=positive(travel_fraction,'travel fraction',zero=True)
    if travel_fraction> .25:raise ValueError('Travel fraction maximum is .25')
    if kinematic_contact_phase not in ('start','end'):raise ValueError('Invalid kinematic contact phase')
    bodies,mass,inertia,axes,feature=prepare(scene)
    if solver=='normal_coupled' and any(b['friction']!=0 or b['restitution']!=0 for b in bodies):raise ValueError('normal_coupled requires zero friction and restitution; use coupled otherwise')
    if position_stabilization not in ('split','velocity_only'):raise ValueError('Invalid position stabilization')
    contact_tolerance_m_s=positive(contact_tolerance_m_s,'contact tolerance')
    contact_slop_m=positive(contact_slop_m,'contact slop',zero=True)
    if contact_slop_m>feature*1e-5:raise ValueError('Contact slop exceeds 1e-5 of smallest feature')
    if solver=='coulomb' and (kinematic_contact_phase!='start' or any(b['restitution']!=0 for b in bodies)):
        raise ValueError('coulomb requires start-phase contacts and zero restitution')
    margin=positive(scene.get('margin_m',0),'margin',zero=True)
    if margin>feature*.1:raise ValueError('Margin exceeds 10% of feature')
    wire=dict(bodies=bodies,gravity=vector(scene.get('gravity',[0,0,-9.81]),3,'gravity').tolist(),frames=frames,dt=dt,primary_steps=primary_steps,iterations=iterations,solver=solver,travel_fraction=travel_fraction,minimum_feature_m=feature,margin_m=margin,kinematic_contact_phase=kinematic_contact_phase,position_stabilization=position_stabilization,preassembly_elimination=preassembly_elimination,contact_tolerance_m_s=contact_tolerance_m_s,contact_slop_m=contact_slop_m,contact_recovery=contact_recovery)
    if rejected_contact_path is not None:
        if solver!='coulomb':raise ValueError('Rejection snapshots require coulomb solver')
        path=Path(rejected_contact_path).expanduser().resolve()
        if path.exists():raise ValueError('Rejection snapshot destination already exists')
        if not path.parent.is_dir():raise ValueError('Rejection snapshot parent must exist')
        wire['rejected_contact_path']=str(path)
    if 'container_interior_half_extents_m' in scene:
        half=vector(scene['container_interior_half_extents_m'],3,'container half extents')
        if np.min(half)<=0 or bodies[0]['type']!='kinematic':raise ValueError('Container monitor requires positive extents and first kinematic body')
        wire.update(container_half=half.tolist(),container_frame_rotation=Rotation.from_matrix(axes[0].T).as_quat().tolist())
    start=time.perf_counter()
    process=subprocess.run([str(binary)],input=json.dumps(wire),text=True,capture_output=True,check=True)
    out=json.loads(process.stdout);out['wall_time_s']=time.perf_counter()-start
    states=np.asarray(out['states'])
    for i,Q in enumerate(axes):
        states[:,i,3:7]=(Rotation.from_quat(states[:,i,3:7])*Rotation.from_matrix(Q.T)).as_quat()
    if not np.isfinite(states).all():raise RuntimeError('Nonfinite 3D state')
    out.update(states=states.tolist(),mass=mass,inertia_body_kg_m2=inertia,body_types=[b['type'] for b in bodies],physical_setup_id=hashlib.sha256(json.dumps(scene,sort_keys=True,separators=(',',':')).encode()).hexdigest(),numerical_model=dict(bullet_commit=BULLET_COMMIT,solver=solver,primary_steps=primary_steps,iterations=iterations,travel_fraction=travel_fraction,margin_m=margin,kinematic_contact_phase=kinematic_contact_phase,position_stabilization=position_stabilization,preassembly_elimination=preassembly_elimination,dantzig_impulse_sanity_limit=1e30,adaptive_policy={'contact_threshold':12,'closing_speed_threshold_m_s':.01,'dwell_updates':24,'fast_iterations':8},contact_tolerance_m_s=contact_tolerance_m_s,contact_slop_m=contact_slop_m,contact_recovery=dict(enabled=solver=='coulomb' and contact_recovery,max_rows=256,max_newton_steps_per_attempt=64,max_svd_calls_total=256,max_jacobi_sweeps_per_svd=64,max_gauge_directions_per_start=4,starts='warm, mechanically neutral gauges, cold',minimum_iteration_budget=64),tangent_gyro_rhs=('consistent free angular velocity' if solver=='coulomb' else 'upstream convention'),friction=('isotropic Coulomb disk; product mixing; residual-gated refinement' if solver=='coulomb' else 'two-direction pyramid; product mixing'),restitution='product mixing; zero velocity threshold'))
    return out


def energy(result):
    s=np.asarray(result['states']); E=np.zeros(len(s))
    for i,m in enumerate(result['mass']):
        if m==0:continue
        R=Rotation.from_quat(s[:,i,3:7]).as_matrix()
        I=np.einsum('tij,jk,tlk->til',R,np.asarray(result['inertia_body_kg_m2'][i]),R)
        E+=m*np.sum(s[:,i,7:10]**2,axis=1)/2+np.einsum('ti,tij,tj->t',s[:,i,10:13],I,s[:,i,10:13])/2
    return E


def errors(reference,candidate):
    if reference['physical_setup_id']!=candidate['physical_setup_id']:raise ValueError('Different physical scenes')
    if reference['mass']!=candidate['mass'] or reference['inertia_body_kg_m2']!=candidate['inertia_body_kg_m2']:raise ValueError('Different mass or inertia')
    if not np.allclose(reference['times'],candidate['times'],atol=1e-12,rtol=0):raise ValueError('Different sample times')
    a=np.asarray(reference['states']);b=np.asarray(candidate['states']);ids=np.flatnonzero(np.asarray(reference['mass'])>0)
    a=a[:,ids];b=b[:,ids]
    angular=(Rotation.from_quat(a[:,:,3:7].reshape(-1,4)).inv()*Rotation.from_quat(b[:,:,3:7].reshape(-1,4))).magnitude()
    rms=lambda v:float(np.sqrt(np.mean(np.sum(v*v,axis=-1))))
    return dict(position_m=rms(a[:,:,:3]-b[:,:,:3]),velocity_m_s=rms(a[:,:,7:10]-b[:,:,7:10]),omega_rad_s=rms(a[:,:,10:13]-b[:,:,10:13]),orientation_rad=float(np.sqrt(np.mean(angular**2))))
