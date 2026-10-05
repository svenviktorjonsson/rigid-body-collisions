"""Independent audit of 72 frozen separated-endpoint diagnostic cases.

Does not import the engine, study runner, or their momentum/energy routines.
No native executable or network is required. Geometry, impulses, state momentum,
kinetic energy and zero controls are recomputed from archived physical inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

DEFAULT=Path(__file__).resolve().parent/'predictive-contact-review'
BACKEND='0f17e758c731da1cf7d32f4e5f7ba6233d539951'
BULLET='2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5'
FROZEN_SOURCE={
    'spatial_backend/coulomb.h':'3187f6e6a356995959c0ff567eb9d46d401d05c4fab68a02f7e1c0319b74a0aa',
    'spatial_backend/normal_qp.h':'1e1c90a45af391d6ec1747a0f6677327f416f2c76c8d120ef76ace7e134bac63',
    'spatial_engine.py':'aa4717c4877b4f02d344a7ccec71396d6366411eed9588e79a45597371937105',
    'research/predictive_contact_harness.cpp':'2802616116bb34e97a6ebd09c98b757f6ee4cb926b8e8fb3e335cad9f4b33fe3'}

def sha(data):return hashlib.sha256(data).hexdigest()

def equal(actual,expected,label,tolerance=2e-10):
    a=np.asarray(actual,dtype=float);e=np.asarray(expected,dtype=float)
    if a.shape!=e.shape or not np.isfinite(a).all() or not np.isfinite(e).all():
        raise ValueError(label+' shape/nonfinite mismatch')
    if np.max(np.abs(a-e),initial=0)>tolerance:
        raise ValueError(label+' differs from independently recomputed value')

def quaternion_matrix(q):
    x,y,z,w=q
    equal(np.dot(q,q),1.,'unit quaternion',2e-12)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def audit(directory=DEFAULT):
    p=Path(directory);read=lambda name:json.loads((p/name).read_text())
    hashes=read('artifact-hashes.json')
    required={'native-harness.json','full-engine.json','full-engine-scenes.json',
              'summary.json','provenance.json','execution-source.zip'}
    if set(hashes)!=required:raise ValueError('Unexpected/missing frozen artifact')
    for name,digest in hashes.items():
        if sha((p/name).read_bytes())!=digest:raise ValueError('Artifact checksum mismatch '+name)
    provenance=read('provenance.json')
    if provenance['backend_commit']!=BACKEND or provenance['bullet_commit']!=BULLET:
        raise ValueError('Different frozen mechanics source')
    physical=provenance['physical_parameters']
    for name,value in {'mass_each_kg':1,'radius_m':.1,'sphere_inertia_kg_m2':.004,
        'box_half_extents_m':[.1]*3,'box_inertia_kg_m2':1/150,
        'velocity_A_m_s':[1,0,-1],'velocity_B_m_s':[-1,0,1],
        'initial_omega_rad_s':[0]*3,'gravity_m_s2':[0]*3,'restitution':0}.items():
        equal(physical[name],value,'physical input '+name)
    for name in ['external_forces','external_torques','rolling_twisting_law']:
        if physical[name] is not False:raise ValueError('Unexpected '+name)
    with zipfile.ZipFile(p/'execution-source.zip') as z:
        for name,digest in FROZEN_SOURCE.items():
            if sha(z.read(name))!=digest:raise ValueError('Frozen diagnostic source changed '+name)
        for name in ['BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.cpp',
                     'BulletDynamics/Dynamics/btDiscreteDynamicsWorld.cpp',
                     'BulletCollision/CollisionDispatch/btSphereSphereCollisionAlgorithm.cpp',
                     'BulletCollision/CollisionDispatch/btConvexConvexAlgorithm.cpp']:
            path='bullet/src/'+name
            if sha(z.read(path))!=provenance['dependency_sha256'][path]:
                raise ValueError('Bullet source checksum '+path)
    native=read('native-harness.json');full=read('full-engine.json');scenes=read('full-engine-scenes.json')
    gaps=[-.001,0,1e-10,1e-6,.001,.01,.025];mus=[0,.4,1]
    expected_native={(g,mu,common) for common in [False,True] for g in gaps for mu in mus}
    actual_native={(r['gap_m'],r['pair_mu'],r['common_point']) for r in native}
    if len(native)!=42 or actual_native!=expected_native:raise ValueError('Native case coverage changed')
    metrics=dict(native_cases=len(native),discovery_cases=len(full),maximum_endpoint_identity_error=0.,
                 maximum_linear_momentum_error=0.,maximum_common_point_angular_error=0.,maximum_energy_increase_J=0.)
    def record(delta,predicted,linear,energy,common):
        metrics['maximum_endpoint_identity_error']=max(metrics['maximum_endpoint_identity_error'],float(np.linalg.norm(np.asarray(delta)-predicted)))
        metrics['maximum_linear_momentum_error']=max(metrics['maximum_linear_momentum_error'],float(np.linalg.norm(linear)))
        if common:metrics['maximum_common_point_angular_error']=max(metrics['maximum_common_point_angular_error'],float(np.linalg.norm(delta)))
        metrics['maximum_energy_increase_J']=max(metrics['maximum_energy_increase_J'],energy[-1]-energy[0])
    for r in native:
        g,mu,common=r['gap_m'],r['pair_mu'],r['common_point'];equal(r['step_s'],.01,'step')
        lever=.1+g/2 if common else .1;kt=2+2*lever*lever/.004
        pn=max(0.,1-g/.02) if g>1e-9 else 1.;pt=-min(2/kt,mu*pn)
        impulse=np.array([pt,0,pn]);point_a=np.array([0,0,0 if common else g/2]);point_b=-point_a
        delta=np.cross(point_a-point_b,impulse)
        energy=[2.,2+np.dot([2,0,-2],impulse)+.5*(kt*pt*pt+2*pn*pn)]
        equal(r['impulse_A_N_s'],impulse,'native analytic impulse')
        equal(r['point_A_m'],point_a,'native endpoint A');equal(r['point_B_m'],point_b,'native endpoint B')
        equal(r['normal_impulse_N_s'],pn,'normal impulse')
        equal(r['omega_A_rad_s'],[0,-lever*pt/.004,0],'A spin');equal(r['omega_B_rad_s'],[0,-lever*pt/.004,0],'B spin')
        equal(r['delta_linear_momentum_kg_m_s'],[0]*3,'linear momentum',1e-12)
        equal(r['delta_angular_momentum_kg_m2_s'],delta,'native endpoint torque',2e-13)
        equal(r['endpoint_predicted_delta_L'],delta,'stored torque prediction',2e-13)
        equal([r['energy_before_J'],r['energy_after_J']],energy,'native energy',2e-12)
        if not 0<=r['residual_m_s']<=1e-10:raise ValueError('Native residual gate')
        record(r['delta_angular_momentum_kg_m2_s'],delta,r['delta_linear_momentum_kg_m_s'],energy,common)
    keys=lambda records:{(r['shape'],r['gap_m'],r['pair_mu']) for r in records}
    expected_full={(s,g,mu) for s in ['sphere','box'] for g in [-.001,0,1e-10,.001,.01] for mu in mus}
    if len(full)!=30 or len(scenes)!=30 or keys(scenes)!=expected_full or keys([r['case'] for r in full])!=expected_full:
        raise ValueError('Full-engine case coverage changed')
    scene_map={(r['shape'],r['gap_m'],r['pair_mu']):r['scene'] for r in scenes}
    for r in full:
        c=r['case'];shape,g,mu=c['shape'],c['gap_m'],c['pair_mu'];scene=scene_map[(shape,g,mu)]
        equal(scene['duration'],.01,'scene duration');equal(scene['gravity'],[0]*3,'scene gravity')
        if len(scene['bodies'])!=2:raise ValueError('Scene bodies')
        rho=1/(4*np.pi*.1**3/3) if shape=='sphere' else 125.
        for i,b in enumerate(scene['bodies']):
            if set(b)!={'position','velocity','friction','restitution','shapes'}:raise ValueError('Unexpected authored body fields')
            sign=1 if i==0 else -1
            equal(b['position'],[0,0,sign*(.1+g/2)],'scene centre')
            equal(b['velocity'],[sign,0,-sign],'scene velocity');equal(b['friction'],mu**.5,'body friction')
            equal(b['restitution'],0,'body restitution')
            if len(b['shapes'])!=1 or b['shapes'][0]['kind']!=shape:raise ValueError('Scene shape')
            geometry=b['shapes'][0];equal(geometry['density'],rho,'density')
            equal(geometry['radius'] if shape=='sphere' else geometry['half_extents'],.1 if shape=='sphere' else [.1]*3,'geometry')
        scene_id=sha(json.dumps(scene,sort_keys=True,separators=(',',':')).encode())
        if r['physical_setup_id']!=scene_id:raise ValueError('Physical setup identity')
        inertia=.004 if shape=='sphere' else 1/150
        equal(r['mass'],[1,1],'mass');equal(r['inertia_body_kg_m2'],[np.eye(3)*inertia]*2,'tensor')
        if r['body_types']!=['dynamic','dynamic']:raise ValueError('Finite dynamic pair')
        s=np.asarray(r['states']);equal(s.shape,[2,2,13],'state shape',0);equal(r['times'],[0,.01],'timestamps')
        if not np.isfinite(s).all():raise ValueError('Nonfinite state')
        pn=1 if g<=0 else 0;kt=2+2*.1**2/inertia;pt=-min(2/kt,mu*pn);impulse=np.array([pt,0,pn])
        for i in range(2):
            sign=1 if i==0 else -1
            equal(s[0,i,:3],[0,0,sign*(.1+g/2)],'initial centre')
            equal(s[0,i,7:10],[sign,0,-sign],'initial velocity');equal(s[0,i,10:13],[0]*3,'initial spin')
            equal(s[-1,i,7:10],s[0,i,7:10]+sign*impulse,'final analytic velocity',2e-8)
            equal(s[-1,i,:3],s[0,i,:3]+.01*s[-1,i,7:10],'force-free integration')
            equal(s[-1,i,10:13],[0,-.1*pt/inertia,0],'final analytic spin',2e-8)
        linear=[];angular=[];energies=[]
        for frame in s:
            P=np.zeros(3);L=np.zeros(3);E=0.
            for b in frame:
                Q=quaternion_matrix(b[3:7]);worldI=Q@(np.eye(3)*inertia)@Q.T;v=b[7:10];w=b[10:13]
                P+=v;L+=np.cross(b[:3],v)+worldI@w;E+=.5*np.dot(v,v)+.5*np.dot(w,worldI@w)
            linear.append(P);angular.append(L);energies.append(E)
        delta=angular[-1]-angular[0];actual_impulse=s[-1,0,7:10]-s[0,0,7:10]
        predicted=np.cross([0,0,g],actual_impulse)
        equal(r['delta_angular_momentum_kg_m2_s'],delta,'measured total angular momentum',2e-12)
        equal(delta,predicted,'full-engine endpoint identity',2e-12)
        equal(r['endpoint_predicted_delta_L'],predicted,'full stored prediction',2e-12)
        equal(r['impulse_A_N_s'],actual_impulse,'full stored impulse',2e-12)
        equal(actual_impulse,impulse,'full analytic impulse',2e-8)
        equal(r['delta_linear_momentum_kg_m_s'],linear[-1]-linear[0],'measured linear momentum',2e-12)
        equal(r['energy_J'],energies,'full energy',2e-12)
        model=r['numerical_model']
        for name,value in {'bullet_commit':BULLET,'solver':'coulomb','primary_steps':1,'iterations':4096,
                           'travel_fraction':0.,'margin_m':0.,'kinematic_contact_phase':'start',
                           'position_stabilization':'velocity_only','contact_tolerance_m_s':1e-8,
                           'contact_slop_m':1e-9,'tangent_gyro_rhs':'consistent free angular velocity'}.items():
            if model[name]!=value:raise ValueError('Different numerical parameter '+name)
        if r['scalar_precision']!='float64':raise ValueError('Different scalar precision')
        if r['collision_updates']!=1 or r['coupled_fallbacks']!=0 or not 0<=r['coulomb_residual_max_m_s']<=1e-8:raise ValueError('Native numerical gate')
        record(delta,predicted,linear[-1]-linear[0],energies,False)
    for name,value in metrics.items():equal(read('summary.json')[name],value,'summary '+name,2e-12)
    if any(metrics[k]>1e-10 for k in metrics if k.startswith('maximum_')):raise ValueError('Physical audit gate')
    return metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=DEFAULT)
    print(json.dumps(audit(parser.parse_args().directory),indent=2))
