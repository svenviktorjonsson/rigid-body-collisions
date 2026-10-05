"""Independent 3D trace/provenance/qualification audit; no native simulation needed."""
import hashlib
import json
from pathlib import Path
import statistics
import zipfile
import numpy as np
from spatial_engine import prepare, BULLET_COMMIT
from research.spatial_metrics import containment

DIRECTORY=Path(__file__).parent/'spatial-validation'
SOURCE_SHA256={'spatial_engine.py': '3465d560a03aa1734052bc0739a7083d73e1e804c40b7f5dc42b1898bc9d4205', 'spatial_backend/runner.cpp': 'adbab5268357d3fe7a57a6daa0b673ec855c8896a0871c0b61730b56730b0b35', 'spatial_backend/CMakeLists.txt': '4c240b6e3ef50613f1317282b058c41b3f9914869972ecc6625f311d2309df3e', 'research/spatial_scenes.py': 'fe678cdb8f96bc2695fd9404ed2dca170b677eccc3878672e5d6a93b7579788e', 'research/spatial_metrics.py': '1e9f162e62e7bf2bd11d585833c7246886b6cceaa01e515008d210525e473277', 'research/run_spatial_study.py': '495190214780fe621ffef00f9e453f6ca8e52ac377609d43399021f8a20fd675', 'research/spatial-validation/plan.json': '412dc7c3c26f6c75d72fe206e13155602570f98808eebeaf14077ae46c8ac100'}


def metrics(a,b):
    assert a['physical_setup_id']==b['physical_setup_id']
    assert a['mass']==b['mass'] and a['inertia_body_kg_m2']==b['inertia_body_kg_m2']
    np.testing.assert_array_equal(a['times'],b['times'])
    ids=np.flatnonzero(np.asarray(a['mass'])>0)
    u=np.asarray(a['states'])[:,ids];v=np.asarray(b['states'])[:,ids]
    assert u.shape==v.shape
    delta=u-v
    rms=lambda x:float(np.sqrt(np.mean(np.sum(x*x,axis=-1))))
    # Independent quaternion product conj(u)*v, geodesic angle in [0,pi].
    qa=u[:,:,3:7];qb=v[:,:,3:7]
    imaginary=qa[:,:,3,None]*qb[:,:,:3]-qb[:,:,3,None]*qa[:,:,:3]-np.cross(qa[:,:,:3],qb[:,:,:3])
    real=qa[:,:,3]*qb[:,:,3]+np.sum(qa[:,:,:3]*qb[:,:,:3],axis=-1)
    angle=2*np.arctan2(np.linalg.norm(imaginary,axis=-1),abs(real))
    return dict(position_m=rms(delta[:,:,:3]),velocity_m_s=rms(delta[:,:,7:10]),omega_rad_s=rms(delta[:,:,10:13]),orientation_rad=float(np.sqrt(np.mean(angle**2))))


def energy_change(scene,r):
    state=np.asarray(r['states']);m=np.asarray(r['mass']);g=np.asarray(scene['gravity']);ends=[]
    from scipy.spatial.transform import Rotation
    for frame in (state[0],state[-1]):
        total=0.
        for i,mass in enumerate(m):
            if mass==0:continue
            rotation=Rotation.from_quat(frame[i,3:7]).as_matrix();tensor=rotation@np.asarray(r['inertia_body_kg_m2'][i])@rotation.T
            total+=.5*mass*np.dot(frame[i,7:10],frame[i,7:10])+.5*frame[i,10:13]@tensor@frame[i,10:13]-mass*np.dot(g,frame[i,:3])
        ends.append(total)
    return ends[1]-ends[0]-r['boundary_work_J']


def norm(metrics,budget):return max(metrics[k]/budget[k] for k in budget)
def close(actual,expected):np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-8)


def audit():
    result=DIRECTORY/'results';data=json.loads((result/'summary.json').read_text());plan=json.loads((DIRECTORY/'plan.json').read_text())
    digest=lambda b:hashlib.sha256(b).hexdigest()
    assert digest((DIRECTORY/'plan.json').read_bytes())==data['plan_sha256']
    for name,sha in data['hashes'].items():assert digest((result/name).read_bytes())==sha
    assert data['bullet_commit']==BULLET_COMMIT
    assert data['execution_source_commit']=='6576bc61852f677a06f6e85d34e6f5b649fb6499'
    with zipfile.ZipFile(result/'execution-source.zip') as z:
        assert z.read('research/spatial-validation/plan.json')==(DIRECTORY/'plan.json').read_bytes()
        cmake=z.read('spatial_backend/CMakeLists.txt').decode()
        assert BULLET_COMMIT in cmake and 'f7c486f10b9f645d0577249d02855e076e8368babaa3d31ced6095e7162baee9' in cmake
        for name in z.namelist():
            assert digest(z.read(name))==SOURCE_SHA256[name],f'Execution source changed: {name}'
    authored=json.loads((result/'scenes.json').read_text())
    expected=set()
    for config in plan['scenes']:
        name=config['id']
        for mode in plan['reference_modes']:expected.add(f'{name}/{mode}/0.json')
        for mode in plan['candidate_modes']:
            for i in range(plan['timing']['candidate_retained_repetitions']):expected.add(f'{name}/{mode}/{i}.json')
    with zipfile.ZipFile(result/'traces.zip') as z:
        assert set(z.namelist())==expected
        traces={k:json.loads(z.read(k)) for k in expected}
    assert len(traces)==data['history_count']==102
    reference_budget={k:v*plan['reference_budget_fraction'] for k,v in plan['candidate_budget'].items()}
    physical={}
    for name,item in authored.items():
        scene=item['scene'];get=lambda mode,i=0:traces[f'{name}/{mode}/{i}.json'];receipt=data['scenes'][name]
        bodies,mass,inertia,_,_=prepare(scene)
        for mode,settings in {**plan['reference_modes'],**plan['candidate_modes']}.items():
            count=3 if mode in plan['candidate_modes'] else 1
            for i in range(count):
                r=get(mode,i);state=np.asarray(r['states'])
                assert state.shape==(round(scene['duration']/plan['dt_s'])+1,len(bodies),13)
                assert np.isfinite(state).all() and np.isfinite(r['boundary_work_J'])
                assert r['scalar_precision']=='float64' and r['numerical_model']['bullet_commit']==BULLET_COMMIT
                for k,v in settings.items():assert r['numerical_model'][k]==v
                assert r['physical_setup_id']==digest(json.dumps(scene,sort_keys=True,separators=(',',':')).encode())
                np.testing.assert_allclose(r['mass'],mass,rtol=1e-13,atol=1e-13)
                np.testing.assert_allclose(r['inertia_body_kg_m2'],inertia,rtol=1e-13,atol=1e-13)
                assert r['collision_updates']==sum(r['updates'])==r['coupled_updates']+r['sequential_updates']
                assert r['coupled_fallbacks']>=0 and r['step_s']>0
                if settings['solver']=='sequential':assert r['coupled_updates']==r['coupled_fallbacks']==0
                if settings['solver']=='coupled':assert r['sequential_updates']==0
                # Exact commanded translation, independent of content motion.
                x=scene['bodies'][0]['position'][0];v=scene['bodies'][0]['velocity'][0];last=0
                for t,frame in zip(r['times'],state):
                    value=x;previous=0;velocity=v
                    for cmd in scene['bodies'][0].get('velocity_schedule',[]):
                        event=cmd['time_s']
                        if event>t:break
                        value+=velocity*(event-previous);previous=event;velocity=cmd.get('velocity',[0,0,0])[0]
                    value+=velocity*(t-previous)
                    close(frame[0,0],value)
                d=dict(quaternion_norm_error=float(np.max(abs(np.linalg.norm(state[:,:,3:7],axis=2)-1))),energy_change_minus_boundary_work_J=energy_change(scene,r),max_contact_penetration_m=r['max_contact_penetration_m'],max_closing_contact_speed_m_s=r['max_closing_contact_speed_m_s'],coupled_fallbacks=r['coupled_fallbacks'])
                if item['half'] is not None:
                    d['sampled_container_surface_excess_m']=containment(scene,r,item['half'])
                    # Internal supports may show a larger transient than samples.
                    assert r['max_container_surface_excess_m']>=d['sampled_container_surface_excess_m']-1e-12
                    d['container_surface_excess_m']=max(d['sampled_container_surface_excess_m'],r['max_container_surface_excess_m'])
                else:d['row_final_velocity_max_error_m_s']=float(np.max(np.linalg.norm(state[-1,1:,7:10]-state[-1,0,7:10],axis=1)))
                limits=plan['physical_checks'];physical[f'{name}/{mode}/{i}']=all(d[k]<=limits[k] for k in limits if k in d)
                archived=receipt['reference_diagnostics'] if mode=='reference' else receipt['candidates'][mode]['diagnostics'] if mode in plan['candidate_modes'] and i==0 else None
                if archived:
                    assert set(d)==set(archived)
                    for k in d:close(d[k],archived[k])
                if count==3:
                    np.testing.assert_array_equal(state,np.asarray(get(mode,0)['states']))
                    warm=data['warmups'][f'{name}/{mode}']['states_sha256']
                    assert warm==digest(json.dumps(r['states'],sort_keys=True,separators=(',',':')).encode())
        edge_pass=True
        for (a,b),saved in zip(plan['reference_edges'],receipt['reference_edges']):
            assert saved['modes']==[a,b];error=metrics(get(b),get(a))
            for k in error:close(error[k],saved['errors'][k])
            n=norm(error,reference_budget);close(n,saved['normalized']);edge_pass &= n<=1
        qualified=bool(edge_pass and physical[f'{name}/reference/0']);assert qualified==receipt['reference_qualified']
        choices=[]
        for mode in plan['candidate_modes']:
            saved=receipt['candidates'][mode];error=metrics(get('reference'),get(mode));n=norm(error,plan['candidate_budget'])
            for k in error:close(error[k],saved['errors'][k])
            close(n,saved['normalized'])
            med=statistics.median(get(mode,i)['step_s'] for i in range(3));close(med,saved['median_step_s'])
            passed=qualified and n<=1 and all(physical[f'{name}/{mode}/{i}'] for i in range(3))
            assert passed==saved['accuracy_qualified']
            for k in ['coupled_updates','sequential_updates','collision_updates']:assert saved[k]==get(mode)[k]
            if passed:choices.append(mode)
        choice=min(choices,key=lambda m:receipt['candidates'][m]['median_step_s']) if choices else None
        assert choice==receipt['choice']
    print(f'3D audit PASS: {len(traces)} histories, {sum(s["reference_qualified"] for s in data["scenes"].values())}/6 qualified references; geometry, quaternions, work, containment, budgets and timing selection agree.')
    return data

if __name__=='__main__':audit()
