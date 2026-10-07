"""Independent provenance, mechanics and qualification audit of fast shaking.

Does not import the production adapter, its errors or its diagnostic helpers.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np

DIRECTORY=Path(__file__).parent/'fast-shake-diagnostic'

def digest(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def read(directory,name):return json.loads((directory/name).read_text())

def rotation(q):
    q=np.asarray(q);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    x,y,z,w=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(q.shape[:-1]+(3,3))

def errors(left,right):
    assert left['physical_setup_id']==right['physical_setup_id'],'Different physics'
    assert left['mass']==right['mass'] and left['inertia_body_kg_m2']==right['inertia_body_kg_m2']
    common=sorted(set(round(t,12) for t in left['times']) & set(round(t,12) for t in right['times']))
    assert len(common)>=13,'Missing common samples'
    def align(result):
        index={round(t,12):i for i,t in enumerate(result['times'])}
        return np.asarray([result['states'][index[t]] for t in common])[:,np.asarray(result['mass'])>0]
    a=align(left);b=align(right)
    qa=a[:,:,3:7];qb=b[:,:,3:7]
    qa=qa/np.linalg.norm(qa,axis=-1,keepdims=True);qb=qb/np.linalg.norm(qb,axis=-1,keepdims=True)
    av=qa[:,:,:3];bv=qb[:,:,:3];aw=qa[:,:,3,None];bw=qb[:,:,3,None]
    vector=aw*bv-bw*av-np.cross(av,bv);scalar=(aw*bw)[:,:,0]+np.sum(av*bv,axis=-1)
    angle=2*np.arctan2(np.linalg.norm(vector,axis=-1),abs(scalar))
    rms=lambda value:float(np.sqrt(np.mean(np.sum(value*value,axis=-1))))
    return dict(position_m=rms(a[:,:,:3]-b[:,:,:3]),velocity_m_s=rms(a[:,:,7:10]-b[:,:,7:10]),
                omega_rad_s=rms(a[:,:,10:13]-b[:,:,10:13]),orientation_rad=float(np.sqrt(np.mean(angle*angle))))

def physical(scene,result):
    states=np.asarray(result['states']);mass=np.asarray(result['mass']);g=np.asarray(scene['gravity'])
    R=rotation(states[:,:,3:7]);I=np.asarray(result['inertia_body_kg_m2']);omega=states[:,:,10:13]
    kinetic=.5*np.sum(mass[None,:,None]*states[:,:,7:10]**2,axis=(1,2))
    body_omega=np.einsum('tijk,tij->tik',R,omega)
    kinetic+=.5*np.einsum('tij,ijk,tik->t',body_omega,I,body_omega)
    potential=-np.einsum('tij,j,i->t',states[:,:,:3],g,mass)
    relative=states[:,1:,:3]-states[:,0:1,:3]
    local=np.einsum('tij,tkj->tki',np.swapaxes(R[:,0],1,2),relative)
    radii=np.asarray([b['shapes'][0]['radius'] for b in scene['bodies'][1:]])
    half=scene['container_interior_half_extents_m'][0]
    surface=float(np.max(abs(local)+radii[None,:,None]-half))
    return dict(quaternion_norm_error=float(np.max(abs(np.linalg.norm(states[:,:,3:7],axis=2)-1))),
                energy_change_minus_boundary_work_J=float(kinetic[-1]+potential[-1]-kinetic[0]-potential[0]-result['boundary_work_J']),
                sampled_container_surface_excess_m=surface,
                container_surface_excess_m=max(surface,result['max_container_surface_excess_m']))

def assert_close(actual,recorded,label):
    for key,value in actual.items():
        assert np.isclose(value,recorded[key],rtol=1e-9,atol=1e-10),label+':'+key

def verify_trial(scene,result,settings,common):
    if 'rejected' in result:
        assert result['rejected'] and result.get('rejection_stage')=='adapter_input_validation'
        return
    state=np.asarray(result['states']);times=np.asarray(result['times'])
    assert state.shape==(round(scene['duration']/settings['dt'])+1,28,13) and np.isfinite(state).all()
    np.testing.assert_allclose(times,np.arange(len(times))*settings['dt'],rtol=0,atol=1e-14)
    assert result['physical_setup_id']==digest(canonical(scene)),'Physics ID'
    np.testing.assert_allclose(state[0,:,:3],[b['position'] for b in scene['bodies']],atol=1e-14,rtol=0,err_msg='Initial position')
    expected_x=np.where(times<=.04,20*times,np.where(times<=.08,1.6-20*times,20*(times-.08)))
    np.testing.assert_allclose(state[:,0,0],expected_x,rtol=0,atol=1e-10,err_msg='Prescribed wall trajectory')
    assert result['mass'][0]==0
    np.testing.assert_allclose(result['mass'][1:],1,atol=1e-14,rtol=0)
    np.testing.assert_allclose(result['inertia_body_kg_m2'][1:],np.tile(np.eye(3)*.004,(27,1,1)),atol=1e-14,rtol=0)
    for key,value in common.items():assert result['numerical_model'][key]==value
    for key in ('primary_steps','travel_fraction'):assert result['numerical_model'][key]==settings[key]
    assert result['scalar_precision']=='float64' and result['coupled_fallbacks']==result['sequential_updates']==0
    assert result['coulomb_residual_max_m_s']<=common['contact_tolerance_m_s']
    assert result['coulomb_sweeps_max']<=common['iterations']
    assert result['collision_updates']==sum(result['updates'])==result['coupled_updates']
    assert result['step_s']>0 and result['mobility_matrix_bytes_max']==8*result['mobility_rows_max']**2

def audit(directory=DIRECTORY,*,check_git=True):
    directory=Path(directory);manifest=read(directory,'artifact-hashes.json')
    for name,sha in manifest.items():assert digest((directory/name).read_bytes())==sha,'Artifact hash: '+name
    provenance=read(directory,'provenance.json')
    with zipfile.ZipFile(directory/'execution-source.zip') as archive:
        for name,sha in provenance['source_hashes'].items():
            source=archive.read(name);assert digest(source)==sha,'Source hash: '+name
            if check_git and name in ('spatial_engine.py','spatial_backend/runner.cpp','spatial_backend/coulomb.h','spatial_backend/normal_qp.h'):
                committed=subprocess.check_output(['git','show',provenance['backend_source_commit']+':'+name],cwd=Path(__file__).parents[1])
                assert committed==source,'Source commit: '+name
    for archive_name,metadata_name,key in [('extension-source.zip','extension-provenance.json','snapshot_sha256'),('correction-source.zip','correction-provenance.json','source_sha256')]:
        with zipfile.ZipFile(directory/archive_name) as archive:
            hashes=read(directory,metadata_name)[key]
            assert set(archive.namelist())==set(hashes)
            for name,sha in hashes.items():assert digest(archive.read(name))==sha
    dependencies=read(directory,'dependencies.json')
    with zipfile.ZipFile(directory/'input-source.zip') as archive:
        for name in archive.namelist():assert digest(archive.read(name))==dependencies['input_hashes'][name]
    authored=read(directory,'scene.json');scene=authored['scene']
    assert digest((directory/'scene.json').read_bytes())==dependencies['scene_sha256']
    base=read(directory,'plan.json');extension=read(directory,'extension-plan.json');correction=read(directory,'fixed-correction-plan.json')
    assert correction['original_extension_plan_sha256']==digest((directory/'extension-plan.json').read_bytes())
    settings=dict(base['runs'],**extension['runs'],**correction['runs']);runs={}
    for lane,control in settings.items():
        runs[lane]=read(directory,lane+'.json');verify_trial(scene,runs[lane],control,base['common'])
    root=Path(__file__).parents[1]
    historical_path='research/spatial-friction/results/traces.zip'
    assert digest((root/historical_path).read_bytes())==dependencies['input_hashes'][historical_path],'Historical traces changed'
    with zipfile.ZipFile(root/historical_path) as archive:
        historical=json.loads(archive.read('fast_shake27_spheres/reference_4.json'))
        preceding=json.loads(archive.read('fast_shake27_spheres/reference_3.json'))
    base_summary=read(directory,'summary.json')['runs']
    for lane in base['runs']:
        assert_close(errors(historical,runs[lane]),base_summary[lane]['error_vs_failed_finest'],'Historical comparison '+lane)
        assert_close(errors(preceding,runs[lane]),base_summary[lane]['error_vs_preceding'],'Preceding comparison '+lane)
    assert sum('rejected' in run for run in runs.values())==1,'Missing retained input rejection'
    for archive_name in ('traces.zip','extension-traces.zip','corrected-extension-traces.zip'):
        with zipfile.ZipFile(directory/archive_name) as archive:
            if archive_name=='traces.zip':expected={name+'.json' for name in base['runs']}
            elif archive_name=='extension-traces.zip':expected={name+'.json' for names in extension['references'].values() for name in names}
            else:expected={name+'.json' for names in correction['references'].values() for name in names}
            assert set(archive.namelist())==expected,'Missing archived lane: '+archive_name
            for name in archive.namelist():assert json.loads(archive.read(name))==runs[name[:-5]],'Raw/archive mismatch: '+name
    corrected=read(directory,'corrected-extension-summary.json')
    decisions={}
    for group,names in correction['references'].items():
        record=corrected[group];passed=True
        for lane in names:
            if 'rejected' in runs[lane]:passed=False;continue
            metrics=physical(scene,runs[lane]);assert_close(metrics,record['physical'][lane]['metrics'],'Physical '+lane)
            eligible=all(metrics[k]<=value for k,value in correction['physical_gates'].items())
            assert eligible==record['physical'][lane]['passed'];passed &= eligible
        for i,(left,right) in enumerate(zip(names[:-1],names[1:])):
            error=errors(runs[left],runs[right]);assert_close(error,record['edges'][i]['error'],'Edge '+left)
            edge_pass=all(error[k]<=value/4 for k,value in correction['trajectory_budget'].items())
            assert edge_pass==record['edges'][i]['passed'];passed &= edge_pass
        assert bool(passed)==record['qualified'],'Qualification '+group;decisions[group]=bool(passed)
    candidate_record=read(directory,'candidate-comparison.json')
    fixed=runs['fixed_1_25us_corrected'];guard=runs['eighth_travel']
    assert_close(errors(fixed,guard),candidate_record['cross_reference_errors'],'Cross reference')
    with zipfile.ZipFile(root/historical_path) as archive:
        for name in ('coarse','medium','fine'):
            outcomes=[]
            for i in range(3):
                candidate=json.loads(archive.read(f'fast_shake27_spheres/{name}_{i}.json'))
                stored=candidate_record[name]['runs'][i];d=physical(scene,candidate)
                e_fixed=errors(candidate,fixed);e_guard=errors(candidate,guard)
                assert_close(d,stored['physical'],'Candidate physical '+name)
                assert_close(e_fixed,stored['errors_vs_fixed'],'Candidate fixed '+name)
                assert_close(e_guard,stored['errors_vs_guard'],'Candidate guard '+name)
                eligible=all(decisions.values()) and all(d[k]<=v for k,v in correction['physical_gates'].items())
                passed=eligible and all(e[k]<=v for e in (e_fixed,e_guard) for k,v in correction['trajectory_budget'].items())
                assert bool(passed)==stored['qualified'],'Candidate qualification '+name;outcomes.append(bool(passed))
            assert all(outcomes)==candidate_record[name]['all_three_qualified']
    assert runs['fixed_1_25us']['rejected']=='Invalid primary_steps'
    print('Fast-shake audit PASS:',len(runs)-1,'histories, 1 retained input rejection; independent reference decisions',decisions)
    return decisions

if __name__=='__main__':audit()
