"""Independent momentum diagnostic, explicitly supplied gap rows vs discovery.

Run from repository root: python -m research.predictive_contact_review.
Uses frozen 0f17 backend/adapter, rather than parent-owned work in progress.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import zipfile
import numpy as np
import scipy
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/predictive-contact-review'
SNAPSHOT = ROOT / 'research/fast-shake-diagnostic/execution-source.zip'
BINARY = Path('/tmp/fast-shake-diagnostic-runner')
BACKEND = '0f17e758c731da1cf7d32f4e5f7ba6233d539951'
BULLET = '2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def momenta(result):
    """Full physical COM orbital + world-inertia spin momentum, fixed origin."""
    states = np.asarray(result['states']); mass = np.asarray(result['mass'])
    P = np.sum(mass[None, :, None] * states[:, :, 7:10], axis=1)
    L = np.zeros((len(states), 3))
    for i, I in enumerate(result['inertia_body_kg_m2']):
        R = Rotation.from_quat(states[:, i, 3:7]).as_matrix()
        world_I = R @ np.asarray(I) @ np.swapaxes(R, -1, -2)
        L += np.cross(states[:, i, :3], mass[i]*states[:, i, 7:10])
        L += np.einsum('nij,nj->ni', world_I, states[:, i, 10:13])
    return P, L

def main():
    OUT.mkdir(exist_ok=True)
    libraries = [ROOT / f'build/spatial/_deps/bullet-build/src/{s}/lib{s}.a'
                 for s in ['BulletDynamics', 'BulletCollision', 'LinearMath']]
    includes = [ROOT/'build/bullet-inspect/src', ROOT/'build/spatial/_deps/json-src/include']
    dependencies = {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in libraries}
    dependencies[str(BINARY)] = sha(BINARY.read_bytes())
    dependencies[str(SNAPSHOT.relative_to(ROOT))] = sha(SNAPSHOT.read_bytes())
    frozen = {}
    with zipfile.ZipFile(SNAPSHOT) as z:
        for n in ['spatial_backend/coulomb.h','spatial_backend/normal_qp.h','spatial_engine.py']:
            frozen[n] = z.read(n)
    for n in ['spatial_backend/coulomb.h','spatial_backend/normal_qp.h','spatial_engine.py']:
        published = subprocess.run(['git','show',BACKEND+':'+n],cwd=ROOT,
                                   capture_output=True,check=True).stdout
        if frozen[n] != published:raise AssertionError('Frozen source mismatch '+n)
    cache=(ROOT/'build/spatial/CMakeCache.txt').read_text()
    source_line=next(line for line in cache.splitlines() if line.startswith('FETCHCONTENT_SOURCE_DIR_BULLET:PATH='))
    source_root=Path(source_line.split('=',1)[1])
    dependencies['build/spatial/CMakeCache.txt']=sha(cache.encode())
    bullet_sources = {}
    for n in ['BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.cpp',
              'BulletDynamics/Dynamics/btDiscreteDynamicsWorld.cpp',
              'BulletCollision/CollisionDispatch/btSphereSphereCollisionAlgorithm.cpp',
              'BulletCollision/CollisionDispatch/btConvexConvexAlgorithm.cpp']:
        inspected=(ROOT/'build/bullet-inspect/src'/n).read_bytes()
        built=(source_root/'src'/n).read_bytes()
        if inspected != built:raise AssertionError('Bullet inspected/build source mismatch '+n)
        bullet_sources['bullet/src/'+n]=built
        dependencies['bullet/src/'+n]=sha(built)
    harness = ROOT/'research/predictive_contact_harness.cpp'
    flags = ['-std=c++17','-O3','-DNDEBUG','-DBT_USE_DOUBLE_PRECISION','-ffp-contract=off']
    with tempfile.TemporaryDirectory(prefix='predictive-contact-') as td:
        work = Path(td)
        for n,data in frozen.items():(work/Path(n).name).write_bytes(data)
        cmd = ['c++',*flags,'-I'+str(work),*['-I'+str(p) for p in includes],str(harness),
               *map(str,libraries),'-o',str(work/'harness')]
        compilation = subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,check=True)
        cases = [dict(gap_m=g,pair_mu=mu,step_s=.01,common_point=common)
                 for common in [False,True]
                 for g in [-.001,0,1e-10,1e-6,.001,.01,.025] for mu in [0,.4,1]]
        result = subprocess.run([str(work/'harness')],input=json.dumps(cases),text=True,
                                capture_output=True,check=True)
        native = json.loads(result.stdout)
        spec = importlib.util.spec_from_file_location('frozen_predictive_adapter',work/'spatial_engine.py')
        adapter = importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        scenes = []; raw = []
        for shape in ['sphere','box']:
            for gap in [-.001,0,1e-10,.001,.01]:
                for mu in [0,.4,1]:
                    bodies=[]
                    for z,v in [(.1+gap/2,[1,0,-1]),(-.1-gap/2,[-1,0,1])]:
                        geometry = (dict(kind='sphere',radius=.1,density=1/(4*np.pi*.1**3/3))
                                    if shape=='sphere' else dict(kind='box',half_extents=[.1]*3,density=125))
                        bodies.append(dict(position=[0,0,z],velocity=v,friction=mu**.5,
                                           restitution=0,shapes=[geometry]))
                    scene=dict(duration=.01,gravity=[0,0,0],bodies=bodies)
                    record=dict(shape=shape,gap_m=gap,pair_mu=mu,scene=scene)
                    scenes.append(record)
                    r=adapter.run(scene,dt=.01,primary_steps=1,iterations=4096,
                        solver='coulomb',travel_fraction=0,kinematic_contact_phase='start',
                        position_stabilization='velocity_only',binary=BINARY)
                    P,L=momenta(r);s=np.asarray(r['states'])
                    impulse=np.asarray(r['mass'])[0]*(s[-1,0,7:10]-s[0,0,7:10])
                    predicted=np.cross([0,0,gap],impulse)
                    r.update(case={k:v for k,v in record.items() if k!='scene'},
                             delta_linear_momentum_kg_m_s=(P[-1]-P[0]).tolist(),
                             delta_angular_momentum_kg_m2_s=(L[-1]-L[0]).tolist(),
                             impulse_A_N_s=impulse.tolist(),endpoint_predicted_delta_L=predicted.tolist(),
                             energy_J=adapter.energy(r).tolist())
                    raw.append(r)
    checks = dict(native_cases=len(native),discovery_cases=len(raw),
                  maximum_endpoint_identity_error=0.,maximum_linear_momentum_error=0.,
                  maximum_common_point_angular_error=0.,maximum_energy_increase_J=0.)
    for r in native+raw:
        delta=np.asarray(r['delta_angular_momentum_kg_m2_s'])
        prediction=np.asarray(r['endpoint_predicted_delta_L'])
        checks['maximum_endpoint_identity_error']=max(checks['maximum_endpoint_identity_error'],float(np.linalg.norm(delta-prediction)))
        checks['maximum_linear_momentum_error']=max(checks['maximum_linear_momentum_error'],float(np.linalg.norm(r['delta_linear_momentum_kg_m_s'])))
        if r.get('common_point',False):checks['maximum_common_point_angular_error']=max(checks['maximum_common_point_angular_error'],float(np.linalg.norm(delta)))
        energy=r.get('energy_J',[r.get('energy_before_J'),r.get('energy_after_J')])
        checks['maximum_energy_increase_J']=max(checks['maximum_energy_increase_J'],energy[-1]-energy[0])
    for key in ['maximum_endpoint_identity_error','maximum_linear_momentum_error',
                'maximum_common_point_angular_error','maximum_energy_increase_J']:
        if checks[key]>1e-10:raise AssertionError((key,checks[key]))
    # Evidence gates: observable leakage plus geometric/friction zero controls.
    leak=[r for r in native if not r['common_point'] and r['gap_m']==.001 and r['pair_mu']==1][0]
    assert abs(leak['delta_angular_momentum_kg_m2_s'][1]+.001/3.5)<1e-13
    assert all(np.linalg.norm(r['delta_angular_momentum_kg_m2_s'])<1e-12 for r in native
               if r['common_point'] or r['gap_m']==0 or r['pair_mu']==0)
    assert all(np.linalg.norm(r['impulse_A_N_s'])==0 for r in raw if r['case']['gap_m']>0)
    provenance=dict(backend_commit=BACKEND,bullet_commit=BULLET,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        compiler=subprocess.run(['c++','--version'],capture_output=True,text=True,check=True).stdout,
        compile_flags=flags,dependency_sha256=dependencies,
        compiler_warnings=compilation.stderr,
        physical_parameters=dict(mass_each_kg=1,radius_m=.1,sphere_inertia_kg_m2=.004,
            box_half_extents_m=[.1]*3,box_inertia_kg_m2=1/150,
            velocity_A_m_s=[1,0,-1],velocity_B_m_s=[-1,0,1],initial_omega_rad_s=[0]*3,
            gravity_m_s2=[0]*3,external_forces=False,external_torques=False,
            restitution=0,rolling_twisting_law=False),
        limitations=['Supplied positive-gap manifold is a solver test, not default CCD discovery.',
            'Common-point rows change contact mobility; this is a harness comparator, not an engine fix.',
            'No claim of calibrated material parameters, long-time convergence, or generic shape accuracy.'])
    outputs={'native-harness.json':native,'full-engine.json':raw,'full-engine-scenes.json':scenes,
             'summary.json':checks,'provenance.json':provenance}
    for name,data in outputs.items():(OUT/name).write_text(json.dumps(data,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'execution-source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in {**frozen,**bullet_sources}.items():z.writestr(name,data)
        for p in [harness,Path(__file__)]:z.write(p,str(p.relative_to(ROOT)))
    files=[p for p in OUT.iterdir() if p.suffix in ['.json','.zip'] and p.name!='artifact-hashes.json']
    (OUT/'artifact-hashes.json').write_text(json.dumps({p.name:sha(p.read_bytes()) for p in sorted(files)},indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
