"""Prospective independent combined hull audit; no production metrics imports.

Historical auditors remain untouched. Portable source/archive checks are default;
current executable, linked libraries and loader checks are explicitly opt-in.
"""
import argparse,hashlib,itertools,json,re,subprocess,zipfile
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'research/hull-combined-completion'
BASELINE_SOURCE='52f7e6d244e92a8405134e0222ce14fc3eda0ef6'
PRECEDING_SOURCE='108a9bb4c7899f75d760b27b179cc56557904a08'
CHANGE={'position_stabilization':{'baseline':'split_translation_gap','candidate':'split_translation_combined'},
        'early_component_recovery':{'baseline':False,'candidate':True},
        'shape_cache_margin_order':{'baseline':'recalc before margin','candidate':'margin before recalc'}}
FIELDS=('translation_pose_ledger_updates','translation_pose_displacement_max_m',
        'translation_pose_potential_change_J','translation_pose_absolute_potential_change_J',
        'translation_pose_orbital_change_kg_m2_s','translation_pose_absolute_orbital_change_kg_m2_s')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(path):return sha(Path(path).read_bytes())
def is_digest(value):return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None
def count(value):assert isinstance(value,int) and not isinstance(value,bool) and value>=0;return value
def save(path,report):
    with Path(path).open('x') as out:out.write(json.dumps(report,indent=2,allow_nan=False)+'\n')

def audit_plan(study=STUDY,require_ready=False):
    study=Path(study);plan=json.loads((study/'plan.json').read_text())
    assert plan['baseline_source_commit']==BASELINE_SOURCE and plan['preceding_protocol_source_commit']==PRECEDING_SOURCE
    assert plan['baseline_plan']=='research/hull-translation-completion/plan.json'
    assert plan['baseline_scenes']=='research/hull-translation-completion/results/scenes.json'
    baseline=json.loads((ROOT/plan['baseline_plan']).read_text())
    expected=dict(baseline['common'],position_stabilization='split_translation_combined',early_component_recovery=True)
    assert plan['common']==expected and plan['declared_numerical_change']==CHANGE
    assert plan['declared_shape_cache_margin_order']==CHANGE['shape_cache_margin_order']
    for key in ('dt_s','trajectory_budget','physical_gates','reference_rule','scenes','candidates','candidate_repetitions','contact_point_policy'):
        assert plan[key]==baseline[key]
    assert len(plan['scenes'])==2 and [c['seed'] for c in plan['scenes']]==[42,7301]
    assert [c['side'] for c in plan['scenes']]==[2,3]
    assert all(c['duration']==.12 and c['fractions']==[.06,.03,.015] for c in plan['scenes'])
    assert plan['common']['contact_tolerance_m_s']==1e-8 and plan['common']['contact_slop_m']==1e-9
    assert plan['native_ledger_fields']==list(FIELDS)
    assert plan['candidates']=={} and plan['candidate_repetitions']==0
    original={plan['baseline_plan'],plan['baseline_scenes'],
              'research/hull-translation-completion/results/traces.zip',
              'research/hull-translation-completion/results/execution-source.zip'}
    assert set(plan['baseline_artifact_hashes'])==original
    hashes=dict(plan['baseline_artifact_hashes'])
    preceding=plan['preceding_protocol_artifact_hashes']
    pending=plan['preceding_protocol_freeze_status']=='PENDING_CURRENT_SIX_ATTEMPT_TERMINAL_ARCHIVE'
    if pending:
        assert not preceding and not require_ready
        assert plan['preparation_status']=='READY_WITH_SCOPE_AWAITING_FINAL_FREEZE'
    else:
        assert plan['preceding_protocol_freeze_status']=='FINALIZED_SIX_ATTEMPT_ARCHIVE'
        assert plan['preparation_status']=='FINALIZED_FOR_ROOT_PUBLISHED_EXECUTION'
        directory='research/hull-gap-completion/results'
        required={directory+'/'+p for p in ('summary.json','scenes.json','traces.zip','execution-source.zip')}
        required.add('research/hull-gap-completion/plan.json')
        assert set(preceding)==required
        terminal=json.loads((ROOT/(directory+'/summary.json')).read_text())
        assert terminal['execution_source_commit']==PRECEDING_SOURCE
        assert terminal['complete'] is True and terminal['attempt_count']==terminal['planned_attempt_count']==6
        hashes.update(preceding)
    assert plan['preceding_protocol_plan']=='research/hull-gap-completion/plan.json'
    previous=json.loads((ROOT/plan['preceding_protocol_plan']).read_text())
    assert previous['common']==dict(baseline['common'],position_stabilization='split_translation_gap')
    for path,expected_hash in hashes.items():assert is_digest(expected_hash) and digest(ROOT/path)==expected_hash
    assert plan['planned_combined_helper']=='spatial_backend/translation_combined.h'
    assert {'spatial_backend/translation_combined.h','research/audit_hull_combined_completion.py'}<=set(plan['freeze_required_paths'])
    return plan,hashes,pending

def library_metadata(libraries):
    assert libraries and all(Path(p).is_absolute() and is_digest(h) for p,h in libraries.items())
    assert any(re.search(r'(ld-linux|ld-musl|ld-[0-9])',Path(p).name) for p in libraries)
    return libraries

def linked_paths(linked):
    return {str(Path(p).resolve()) for p in re.findall(r'(/\S+)\s+\(',linked)}


def expected_sources(source,plan):
    fixed={'spatial_engine.py','spatial_fidelity.py','research/spatial_scenes.py','research/spatial_metrics.py',
           'spatial_backend/runner.cpp','spatial_backend/CMakeLists.txt',
           'research/hull-combined-completion/runner.py','research/hull-combined-completion/plan.json',
           'research/hull-combined-completion/README.md','research/audit_shared_hulls.py',
           'research/audit_hull_active_completion.py'}
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',source,'spatial_backend'],cwd=ROOT,text=True).splitlines()
    native={p for p in paths if Path(p).suffix in ('.h','.cpp')}
    return fixed|native|set(plan['freeze_required_paths'])


def runtime_metadata(study,source):
    study=Path(study);plan,hashes,pending=audit_plan(study,require_ready=True)
    assert not pending and re.fullmatch(r'[0-9a-f]{40}',source)
    directory=study/'results';provenance=json.loads((directory/'checkpoints/provenance.json').read_text())
    assert provenance['execution_source_commit']==source and provenance['plan_sha256']==digest(study/'plan.json')
    assert provenance['baseline_artifact_hashes']==hashes and provenance['declared_numerical_change']==CHANGE
    assert provenance['preceding_protocol_source_commit']==PRECEDING_SOURCE
    assert is_digest(provenance['binary_sha256'])
    libraries=provenance['runtime_library_hashes']
    library_metadata(libraries)  # Loader omissions are not inherited from historical manifests.
    assert provenance['thread_environment']==plan['thread_environment']
    with zipfile.ZipFile(directory/'execution-source.zip') as archive:
        assert len(archive.namelist())==len(set(archive.namelist()))
        assert set(archive.namelist())==set(provenance['source_hashes'])==expected_sources(source,plan)
        for name,expected in provenance['source_hashes'].items():
            content=archive.read(name);assert is_digest(expected) and sha(content)==expected
            assert content==subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)
        required={'spatial_engine.py','spatial_backend/runner.cpp','spatial_backend/translation_combined.h',
                  'research/audit_hull_combined_completion.py',
                  'research/hull-combined-completion/runner.py','research/hull-combined-completion/plan.json',
                  'research/hull-combined-completion/auditor_checks.py'}
        assert required<=set(archive.namelist())
        assert archive.read('research/hull-combined-completion/plan.json')==(study/'plan.json').read_bytes()
        assert archive.read('research/audit_hull_combined_completion.py')==Path(__file__).read_bytes()
    summary_path=directory/'summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        for key in ('execution_source_commit','plan_sha256','binary_sha256','runtime_library_hashes','source_hashes','baseline_artifact_hashes','declared_numerical_change'):
            assert summary[key]==provenance[key]
    return provenance,plan

def audit_runtime(study,source):
    provenance,_=runtime_metadata(study,source)
    binary=ROOT/'build/spatial/spatial_runner';assert digest(binary)==provenance['binary_sha256']
    linked=subprocess.check_output(['ldd',str(binary)],text=True)
    resolved=linked_paths(linked)
    assert resolved==set(provenance['runtime_library_hashes'])
    for path,expected in provenance['runtime_library_hashes'].items():
        assert str(Path(path).resolve())==path and digest(path)==expected
    return dict(schema='combined-current-runtime-attestation-v1',execution_source_commit=source,
                current_host_runtime_checked=True,current_runtime_integrity_passed=True,
                binary_sha256=provenance['binary_sha256'],runtime_library_hashes=provenance['runtime_library_hashes'],trajectory_qualified=False)

def rotation(q):
 q=np.asarray(q);q=q/np.linalg.norm(q);v=q[:3];w=q[3]
 K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
 return (w*w-v@v)*np.eye(3)+2*np.outer(v,v)+2*w*K

def shape_integrals(shape):
 kind=shape['kind']
 if kind=='sphere':
  r=shape['radius'];V=4*np.pi*r**3/3;return V,np.zeros(3),2*V*r*r/5*np.eye(3)
 if kind=='box':
  h=np.asarray(shape['half_extents']);V=8*np.prod(h);return V,np.zeros(3),V/3*np.diag(np.sum(h*h)-h*h)
 P=np.asarray(shape['vertices']);hull=ConvexHull(P);V=0.;first=np.zeros(3);second=np.zeros((3,3))
 for face,equation in zip(hull.simplices,hull.equations):
  a,b,c=P[face]
  if np.dot(np.cross(b-a,c-a),equation[:3])<0:b,c=c,b
  volume=np.dot(a,np.cross(b,c))/6;vertices=np.asarray([a,b,c]);total=vertices.sum(axis=0)
  V+=volume;first+=volume*total/4;second+=volume*(np.outer(total,total)+vertices.T@vertices)/20
 center=first/V;second-=V*np.outer(center,center)
 return V,center,np.trace(second)*np.eye(3)-second

def body_integrals(body):
 parts=[]
 for s in body['shapes']:
  V,c,I=shape_integrals(s);rho=s.get('density',1.);R=rotation(s.get('orientation',[0,0,0,1]));c=R@c+np.asarray(s.get('center',[0,0,0]));parts.append((V*rho,c,rho*R@I@R.T))
 mass=sum(p[0] for p in parts);center=sum(m*c for m,c,I in parts)/mass;inertia=np.zeros((3,3))
 for m,c,I in parts:
  d=c-center;inertia+=I+m*((d@d)*np.eye(3)-np.outer(d,d))
 return mass,center,inertia

def trajectory_metrics(scene,result):
 states=np.asarray(result['states']);assert states.ndim==3 and states.shape[2]==13 and np.isfinite(states).all()
 g=np.asarray(scene.get('gravity',[0,0,-9.81]));E=np.zeros(len(states));sampled=-np.inf;half=scene['container_interior_half_extents_m'][0]
 for i,body in enumerate(scene['bodies']):
  geometric_mass,center,I=body_integrals(body);m=geometric_mass if body.get('type','dynamic')=='dynamic' else 0.
  assert np.isclose(result['mass'][i],m,rtol=1e-12,atol=1e-12)
  assert np.allclose(result['inertia_body_kg_m2'][i],I,rtol=1e-11,atol=1e-11)
  for j,frame in enumerate(states):
   x=frame[i];R=rotation(x[3:7]);spin=R.T@x[10:13]
   if m:E[j]+=.5*m*(x[7:10]@x[7:10])+.5*spin@I@spin-m*g@x[:3]
   if i==0:continue
   Rc=rotation(frame[0,3:7]);relative=Rc.T@(x[:3]-frame[0,:3])
   for s in body['shapes']:
    Rs=rotation(s.get('orientation',[0,0,0,1]));offset=np.asarray(s.get('center',[0,0,0]))-center
    if s['kind']=='sphere':excess=np.max(abs(relative+Rc.T@R@offset))+s['radius']-half
    else:
     vertices=np.asarray(s['vertices']) if s['kind']=='hull' else np.asarray(list(itertools.product([-1,1],repeat=3)))*s['half_extents']
     points=(vertices@Rs.T+offset)@R.T@Rc+relative
     excess=np.max(abs(points))-half+(scene.get('margin_m',0) if s['kind']=='hull' else 0)
    sampled=max(sampled,float(excess))
 return dict(quaternion_norm_error=float(np.max(abs(np.linalg.norm(states[:,:,3:7],axis=2)-1))),
             energy_change_minus_boundary_work_J=float(E[-1]-E[0]-result['boundary_work_J']),
             container_surface_excess_m=max(sampled,result['max_container_surface_excess_m']),
             sampled_container_surface_excess_m=sampled)

def trajectory_error(left,right):
 assert left['physical_setup_id']==right['physical_setup_id'] and left['mass']==right['mass'] and left['inertia_body_kg_m2']==right['inertia_body_kg_m2']
 assert np.allclose(left['times'],right['times'],atol=1e-12,rtol=0)
 ids=np.flatnonzero(np.asarray(left['mass'])>0);a=np.asarray(left['states'])[:,ids];b=np.asarray(right['states'])[:,ids]
 rms=lambda x:float(np.sqrt(np.mean(np.sum(x*x,axis=-1))))
 qa=a[:,:,3:7];qb=b[:,:,3:7];qa=qa/np.linalg.norm(qa,axis=-1,keepdims=True);qb=qb/np.linalg.norm(qb,axis=-1,keepdims=True)
 vec=qa[:,:,3,None]*qb[:,:,:3]-qb[:,:,3,None]*qa[:,:,:3]-np.cross(qa[:,:,:3],qb[:,:,:3]);scalar=np.abs(np.sum(qa*qb,axis=-1));angle=2*np.arctan2(np.linalg.norm(vec,axis=-1),scalar)
 return dict(position_m=rms(a[:,:,:3]-b[:,:,:3]),velocity_m_s=rms(a[:,:,7:10]-b[:,:,7:10]),omega_rad_s=rms(a[:,:,10:13]-b[:,:,10:13]),orientation_rad=float(np.sqrt(np.mean(angle*angle))))

def authored_progress(scene,progress,dt):
    assert progress['schema']=='native-spatial-progress-v1'
    assert progress['orientation_frame']=='backend principal inertia axes'
    backend=np.asarray(progress['states_backend'],dtype=float);wire=progress['wire_bodies'];times=np.asarray(progress['times'])
    expected=round(scene['duration']/dt);completed=progress['completed_output_frames']
    assert backend.shape==(completed+1,len(scene['bodies']),13) and np.isfinite(backend).all()
    assert 0<=completed<=expected==progress['expected_output_frames']
    assert progress['complete']==(completed==expected)
    np.testing.assert_allclose(times,np.arange(completed+1)*dt,rtol=0,atol=1e-12)
    assert len(wire)==len(scene['bodies'])
    authored=backend.copy();mass=[];inertia=[];certificates=[];native_energy=np.zeros(len(times));g=np.asarray(scene.get('gravity',[0,0,-9.81]))
    for i,body in enumerate(scene['bodies']):
        geometric_mass,center,I=body_integrals(body);m=geometric_mass if body.get('type','dynamic')=='dynamic' else 0.
        mass.append(m);inertia.append(I.tolist());w=wire[i]
        assert w['type']==body.get('type','dynamic') and np.isclose(w['mass'],m,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(backend[0,i,:3],body.get('position',[0,0,0]),rtol=0,atol=1e-12)
        np.testing.assert_allclose(backend[0,i,7:10],body.get('velocity',[0,0,0]),rtol=0,atol=1e-12)
        np.testing.assert_allclose(backend[0,i,10:13],body.get('omega',[0,0,0]),rtol=0,atol=1e-12)
        R0=rotation(body.get('orientation',[0,0,0,1]));P0=rotation(backend[0,i,3:7]);Q=R0.T@P0
        np.testing.assert_allclose(Q.T@Q,np.eye(3),rtol=0,atol=1e-12);assert abs(np.linalg.det(Q)-1)<1e-12
        diagonal=np.asarray(w['principal_inertia']);defect=0.
        if m:
            defect=float(np.max(abs(Q@np.diag(diagonal)@Q.T-I)))
            np.testing.assert_allclose(Q@np.diag(diagonal)@Q.T,I,rtol=1e-11,atol=1e-11)
        else:np.testing.assert_array_equal(diagonal,[0,0,0])
        assert len(w['shapes'])==len(body['shapes'])
        for shape,native in zip(body['shapes'],w['shapes']):
            assert shape['kind']==native['kind']
            for key in ('radius','half_extents','vertices'):
                if key in shape:np.testing.assert_array_equal(shape[key],native[key])
            Rs=rotation(shape.get('orientation',[0,0,0,1]));offset=np.asarray(shape.get('center',[0,0,0]))-center
            np.testing.assert_allclose(native['center'],Q.T@offset,rtol=0,atol=1e-12)
            np.testing.assert_allclose(rotation(native['orientation']),Q.T@Rs,rtol=0,atol=1e-12)
        for key,default in [('friction',.5),('restitution',0)]:assert w[key]==body.get(key,default)
        for j in range(len(times)):
            P=rotation(backend[j,i,3:7]);R=P@Q.T;authored[j,i,3:7]=Rotation.from_matrix(R).as_quat()
            if m:
                spin=P.T@backend[j,i,10:13];v=backend[j,i,7:10]
                native_energy[j]+=.5*m*(v@v)+.5*np.sum(diagonal*spin*spin)-m*g@backend[j,i,:3]
        certificates.append(dict(body=i,principal_to_authored_basis=Q.tolist(),independent_full_tensor_defect_kg_m2=defect))
    result=dict(states=authored.tolist(),times=times.tolist(),mass=mass,inertia_body_kg_m2=inertia,
                boundary_work_J=progress['boundary_work_J'],max_container_surface_excess_m=progress['max_container_surface_excess_m'])
    metrics=trajectory_metrics(scene,result)
    direct=float(native_energy[-1]-native_energy[0]-progress['boundary_work_J'])
    np.testing.assert_allclose(metrics['energy_change_minus_boundary_work_J'],direct,rtol=1e-10,atol=1e-10)
    raw_qerror=float(np.max(abs(np.linalg.norm(backend[:,:,3:7],axis=2)-1)))
    # Normalizing a quaternion for geometry must not hide a malformed raw state.
    metrics['quaternion_norm_error']=max(metrics['quaternion_norm_error'],raw_qerror)
    return result,metrics,certificates


def ledger(record):
    """Validate finite accounting and signed-versus-absolute triangle bounds."""
    updates = record[FIELDS[0]]
    assert isinstance(updates, int) and not isinstance(updates, bool) and updates >= 0
    displacement, potential, absolute_potential = (record[key] for key in FIELDS[1:4])
    orbital = np.asarray(record[FIELDS[4]], dtype=float)
    absolute_orbital = record[FIELDS[5]]
    scalars = np.asarray([displacement, potential, absolute_potential, absolute_orbital], dtype=float)
    assert scalars.shape == (4,) and np.isfinite(scalars).all()
    assert orbital.shape == (3,) and np.isfinite(orbital).all()
    assert displacement >= 0 and absolute_potential >= 0 and absolute_orbital >= 0
    # Floating summation roundoff only; these checks do not loosen a physics gate.
    assert abs(potential) <= absolute_potential + 1e-12 * max(1., absolute_potential)
    orbital_norm = float(np.hypot.reduce(orbital))
    assert np.isfinite(orbital_norm)
    assert orbital_norm <= absolute_orbital + 1e-12 * max(1., absolute_orbital)
    if updates == 0:
        assert np.all(scalars == 0) and np.all(orbital == 0)
    return {key: record[key] for key in FIELDS}


def position_projection(record, tolerance):
    solves = record['translation_split_solves']
    residual = record['translation_split_residual_max_m_s']
    assert isinstance(solves, int) and not isinstance(solves, bool) and solves >= 0
    assert np.isfinite(residual) and 0 <= residual <= tolerance
    if solves == 0:
        assert residual == 0
    return dict(solves=solves, residual_max_m_s=residual, unchanged_tolerance_m_s=tolerance,
                strict_position_projection_gate_passed=True)


def validate_model(result,plan,fraction):
    model=result['numerical_model']
    assert result['contact_point_policy']==model['contact_point_policy']=='shared'
    assert model['shape_cache_margin_order']=='margin before recalc'
    assert model['early_component_recovery'] is True
    for key,value in plan['common'].items():
        if key=='contact_recovery':assert model[key]['enabled'] is value
        else:assert model[key]==value
    assert model['travel_fraction']==fraction and result['scalar_precision']=='float64'


def rejection_capture(snapshot,result,plan):
    """New position rejects MUST contain the pure-normal captured system."""
    phase=snapshot['phase'];n=len(snapshot['b']);assert n>0
    assert snapshot['tolerance_m_s']==plan['common']['contact_tolerance_m_s']
    assert np.isfinite(snapshot['residual_m_s']) and snapshot['residual_m_s']>snapshot['tolerance_m_s']
    assert np.isfinite(snapshot['internal_dt_s']) and snapshot['internal_dt_s']>0
    assert count(snapshot['iteration_budget'])==plan['common']['iterations']
    A=np.asarray(snapshot['A'],float);assert A.shape==(n,n) and np.isfinite(A).all()
    assert np.allclose(A,A.T,rtol=1e-12,atol=1e-12)
    vectors={key:np.asarray(snapshot[key],float) for key in ('b','p','lo','hi')}
    assert all(v.shape==(n,) and np.isfinite(v).all() for v in vectors.values())
    deps=snapshot['dependencies'];assert len(deps)==n and all(isinstance(i,int) and not isinstance(i,bool) for i in deps)
    if phase=='position_translation':
        assert snapshot['schema']=='normal-only-position-rejection-v1'
        assert result['rejected']=='Translation-only position projection failed; repair initial overlap or refine timestep'
        assert deps==[-1]*n and np.all(vectors['lo']==0) and np.all(vectors['hi']>0)
        assert np.all(np.diag(A)>0)
        w=A@vectors['p']-vectors['b']
        independent=float(np.max(abs(vectors['p']-np.maximum(0,vectors['p']-w/np.diag(A)))*np.diag(A)))
        assert np.isfinite(independent)
        assert np.isclose(independent,snapshot['residual_m_s'],rtol=1e-7,atol=1e-10)
    else:
        assert phase=='velocity' and snapshot['schema']=='circular-coulomb-rejection-v1'
        assert result['rejected'].startswith('Coulomb residual gate failed') and 'no friction-law fallback' in result['rejected']
        normals=[i for i,d in enumerate(deps) if d<0];assert normals
        covered=set()
        for i in normals:
            tangents=[j for j,d in enumerate(deps) if d==i]
            assert len(tangents)==2 and vectors['lo'][i]==0 and A[i,i]>0
            t,s=tangents
            assert vectors['hi'][t]==vectors['hi'][s]>=0
            assert vectors['lo'][t]==-vectors['hi'][t] and vectors['lo'][s]==-vectors['hi'][s]
            assert A[t,t]>0 and A[s,s]>0 and A[t,t]*A[s,s]-A[t,s]**2>0
            covered.update((i,t,s))
        assert covered==set(range(n))
    return dict(phase=phase,rows=n,residual_m_s=snapshot['residual_m_s'],matrix_snapshot_available=True)


def ledger_agreement(prefix,final,saved):
    prefix_values=ledger(prefix) if prefix is not None else {}
    final_values=ledger(final) if final is not None else {}
    if final is not None:assert prefix is not None and final_values==prefix_values
    assert saved['native_prefix']==prefix_values and saved['final']==final_values
    assert saved['prefix_available']==(prefix is not None) and saved['final_available']==(final is not None)
    return dict(prefix=prefix_values,final=final_values)


def prefix_record(scene,progress,plan):
    count(progress['completed_output_frames']);count(progress['expected_output_frames']);count(progress['collision_updates'])
    assert isinstance(progress['complete'],bool)
    assert np.isfinite(progress['boundary_work_J']) and np.isfinite(progress['max_container_surface_excess_m'])
    assert np.isfinite(progress['coulomb_residual_max_m_s']) and 0<=progress['coulomb_residual_max_m_s']<=plan['common']['contact_tolerance_m_s']
    result,metrics,certificates=authored_progress(scene,progress,plan['dt_s'])
    accounting=ledger(progress);projection=position_projection(progress,plan['common']['contact_tolerance_m_s'])
    assert all(np.isfinite(value) for value in metrics.values())
    return result,dict(complete=progress['complete'],completed_output_frames=progress['completed_output_frames'],
                      expected_output_frames=progress['expected_output_frames'],collision_updates=progress['collision_updates'],
                      basis_certificates=certificates,diagnostics=metrics,native_prefix=accounting,
                      position_projection=projection,prefix_observation_only=not progress['complete'],trajectory_qualified=False)


def refinement_receipt(plan,runs,eligible):
    edges=[]
    for left,right in [('reference_0','reference_1'),('reference_1','reference_2')]:
        error=trajectory_error(runs[left],runs[right]) if eligible.get(left) and eligible.get(right) else None
        passed=bool(error is not None and all(np.isfinite(error[k]) and error[k]<=limit/4 for k,limit in plan['trajectory_budget'].items()))
        edges.append(dict(left=left,right=right,error=error,passed=passed))
    return dict(reference_qualified=all(e['passed'] for e in edges),physical_eligible=eligible,edges=edges)


def audit_evidence(study,source,require_all=True):
    """Read actual outcomes; rejects/partial observations never qualify."""
    study=Path(study);directory=study/'results';provenance,plan=runtime_metadata(study,source)
    summary=json.loads((directory/'summary.json').read_text());scenes=json.loads((directory/'scenes.json').read_text())
    assert scenes==json.loads((ROOT/plan['baseline_scenes']).read_text())
    for key,expected_hash in summary['hashes'].items():assert digest(directory/key)==expected_hash
    expected={f'{c["id"]}/reference_{i}.json' for c in plan['scenes'] for i in range(3)}
    assert count(summary['attempt_count'])<=6 and summary['planned_attempt_count']==6
    assert count(summary['history_count'])<=summary['attempt_count']
    if require_all:assert summary['complete'] is True and summary['attempt_count']==6 and count(summary['interruption_count'])==0
    histories=0;rejections=[];prefixes=[];receipts={};seen=set()
    with zipfile.ZipFile(directory/'traces.zip') as archive:
        assert len(archive.namelist())==len(set(archive.namelist()))
        assert set(archive.namelist())<=expected
        if require_all:assert set(archive.namelist())==expected
        for config in plan['scenes']:
            name=config['id'];scene=scenes[name]['scene'];runs={};eligible={};physical={}
            for i,fraction in enumerate(config['fractions']):
                lane=f'reference_{i}';key=f'{name}/{lane}.json';progress_path=directory/'progress'/key
                checkpoint=directory/'checkpoints'/key
                prefix_result=None;prefix=None;progress=None
                if progress_path.exists():
                    progress=json.loads(progress_path.read_text());prefix_result,prefix=prefix_record(scene,progress,plan)
                    prefix.update(scene=name,lane=lane,fraction=fraction,path='progress/'+key,sha256=digest(progress_path))
                    prefixes.append(prefix)
                if key not in archive.namelist():
                    assert not require_all and not checkpoint.exists()
                    eligible[lane]=False;continue
                seen.add(key);result=json.loads(archive.read(key));runs[lane]=result
                assert result==json.loads(checkpoint.read_text())
                if require_all:assert progress is not None
                if progress is not None:assert result['native_progress']==dict(path='progress/'+key,sha256=digest(progress_path))
                started=json.loads((directory/'attempts'/key).read_text())
                assert started['status']=='started' and started['scene']==name and started['lane']==lane
                assert started['travel_fraction']==fraction and started['execution_source_commit']==source
                count(started['runner_pid'])
                final_ledger=None
                if 'rejected' in result:
                    eligible[lane]=False
                    assert result['attempt_status'] in ('engine_rejected','interrupted')
                    assert np.isfinite(result['elapsed_s']) and result['elapsed_s']>0
                    if result['attempt_status']=='interrupted':
                        assert not require_all and result['complete'] is False
                        rejections.append(dict(scene=name,lane=lane,interrupted=True,trajectory_qualified=False))
                    else:
                        assert result['exit_code']==1
                        assert result['rejection_dump_status']=='captured' and result['rejection_dump'] is not None
                        diagnostic=summary['rejection_diagnostics'][key];dump=directory/result['rejection_dump']
                        assert str(dump.relative_to(directory))==diagnostic['path'] and digest(dump)==diagnostic['sha256']
                        evidence=rejection_capture(json.loads(dump.read_text()),result,plan)
                        assert evidence['phase']==diagnostic['phase'] and evidence['rows']==diagnostic['rows']
                        assert evidence['residual_m_s']==diagnostic['residual_m_s']
                        rejections.append(dict(scene=name,lane=lane,fraction=fraction,reason=result['rejected'],trajectory_qualified=False,**evidence))
                else:
                    assert result['attempt_status']=='history_complete' and progress['complete'] is True
                    validate_model(result,plan,fraction)
                    expected_times=np.arange(round(scene['duration']/plan['dt_s'])+1)*plan['dt_s']
                    assert np.asarray(result['states']).shape==(len(expected_times),len(scene['bodies']),13)
                    np.testing.assert_allclose(result['times'],expected_times,rtol=0,atol=1e-12)
                    assert result['physical_setup_id']==sha(json.dumps(scene,sort_keys=True,separators=(',',':')).encode()) or result['physical_setup_id']==sha(json.dumps(scene,sort_keys=True).encode())
                    assert np.isfinite(result['boundary_work_J']) and np.isfinite(result['max_container_surface_excess_m'])
                    assert np.isfinite(result['coulomb_residual_max_m_s']) and 0<=result['coulomb_residual_max_m_s']<=plan['common']['contact_tolerance_m_s']
                    a=np.asarray(prefix_result['states']);b=np.asarray(result['states'])
                    np.testing.assert_allclose(a[:,:,:3],b[:,:,:3],rtol=0,atol=1e-12)
                    np.testing.assert_allclose(a[:,:,7:13],b[:,:,7:13],rtol=0,atol=1e-12)
                    np.testing.assert_allclose(abs(np.sum(a[:,:,3:7]*b[:,:,3:7],axis=-1)),1,rtol=0,atol=1e-12)
                    assert result['boundary_work_J']==progress['boundary_work_J']
                    assert result['coulomb_residual_max_m_s']==progress['coulomb_residual_max_m_s']
                    d=trajectory_metrics(scene,result);physical[lane]=d
                    assert all(np.isfinite(value) for value in d.values())
                    eligible[lane]=all(d[k]<=limit for k,limit in plan['physical_gates'].items())
                    final_ledger=ledger(result);assert final_ledger==prefix['native_prefix']
                    assert position_projection(result,plan['common']['contact_tolerance_m_s'])==prefix['position_projection']
                    histories+=1
                ledger_path=directory/'ledgers'/key
                if require_all:assert ledger_path.exists() and prefix is not None
                if ledger_path.exists():
                    saved=json.loads(ledger_path.read_text());meta=summary['native_ledger_receipts'][key]
                    assert meta==dict(path='ledgers/'+key,sha256=digest(ledger_path))
                    ledger_agreement(progress,result if final_ledger is not None else None,saved)
                    assert saved['scope']==plan['ledger_scope']
            receipt=refinement_receipt(plan,runs,eligible);receipt['diagnostics']=physical
            if name in summary['scenes']:
                saved=summary['scenes'][name]
                assert saved['reference_qualified']==receipt['reference_qualified']
                assert saved['choice'] is None and saved['candidates']=={}
                assert saved['physical_eligible']==eligible
                assert len(saved['edges'])==2
                for actual,recorded in zip(receipt['edges'],saved['edges']):
                    assert actual['left']==recorded['left'] and actual['right']==recorded['right']
                    assert actual['passed']==recorded['passed'] and (actual['error'] is None)==(recorded['error'] is None)
                    if actual['error']:
                        assert all(np.isclose(actual['error'][k],recorded['error'][k],rtol=1e-9,atol=1e-10) for k in actual['error'])
            elif require_all:raise AssertionError('Missing terminal scene receipt')
            receipts[name]=receipt
    assert len(seen)==summary['attempt_count'] and histories==summary['history_count']
    if require_all:assert len(prefixes)==6 and len(summary['native_ledger_receipts'])==6
    return dict(schema='independent-combined-hull-audit-v1',execution_source_commit=source,
                declared_numerical_change=CHANGE,complete_attempt_audit=require_all,attempt_count=len(seen),
                history_count=histories,rejection_count=len(rejections),rejections=rejections,
                prefix_count=len(prefixes),prefixes=prefixes,scenes=receipts,
                current_host_runtime_checked=False,
                scope='Independent authored/principal frame conversion, full-tensor kinetic plus gravitational energy minus recorded boundary work, authored geometry containment, strict contact/position residuals and finite signed/absolute pose ledgers. Per-contact wall work and per-update repair impulses are not reconstructed. Only complete histories passing physical gates and BOTH quarter-budget edges qualify; no causal attribution or speed ranking.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=STUDY)
    parser.add_argument('--source-commit');parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--runtime-metadata-only',action='store_true');parser.add_argument('--check-current-runtime',action='store_true')
    parser.add_argument('--progress-only',action='store_true');parser.add_argument('--receipt-prefix',default='independent')
    args=parser.parse_args();study=args.directory.resolve();plan,_,pending=audit_plan(study)
    if args.plan_only:
        print('PENDING preceding final-six archive/source freeze; prospective original52 gates and THREE declarations PASS; no trajectory qualification' if pending else 'Finalized combined plan PASS; no trajectory qualification')
        return
    assert args.source_commit and re.fullmatch(r'[0-9a-f]{40}',args.source_commit)
    assert re.fullmatch(r'[a-zA-Z0-9_-]+',args.receipt_prefix)
    provenance,_=runtime_metadata(study,args.source_commit)
    report=dict(schema='combined-portable-runtime-metadata-v1',execution_source_commit=args.source_commit,
                plan_sha256=provenance['plan_sha256'],binary_sha256=provenance['binary_sha256'],
                runtime_library_hashes=provenance['runtime_library_hashes'],current_host_runtime_checked=False,
                portable_metadata_integrity_passed=True,trajectory_qualified=False)
    save(study/(args.receipt_prefix+'-runtime-metadata-audit.json'),report)
    if args.check_current_runtime:save(study/(args.receipt_prefix+'-current-runtime-audit.json'),audit_runtime(study,args.source_commit))
    if not args.runtime_metadata_only:save(study/(args.receipt_prefix+'-audit.json'),audit_evidence(study,args.source_commit,require_all=not args.progress_only))
    print('Combined archive audit PASS; prefixes/rejects never establish complete accuracy')

if __name__=='__main__':main()
