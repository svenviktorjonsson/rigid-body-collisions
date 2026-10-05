"""Independent analytic and identical-algorithm 3D normal-contact evidence audit."""
import hashlib
import json
from pathlib import Path
import statistics
import zipfile
import numpy as np
from research.spatial_metrics import containment

DIRECTORY=Path(__file__).parent/'spatial-normal'


def audit():
    directory=DIRECTORY/'results';data=json.loads((directory/'summary.json').read_text());plan=json.loads((DIRECTORY/'plan.json').read_text())
    digest=lambda b:hashlib.sha256(b).hexdigest()
    assert data['execution_source_commit']=='76e44754c09c36112e161441462ac5cd34687a49'
    assert digest((DIRECTORY/'plan.json').read_bytes())==data['plan_sha256']
    for name,sha in data['hashes'].items():assert digest((directory/name).read_bytes())==sha
    with zipfile.ZipFile(directory/'execution-source.zip') as z:
        assert set(z.namelist())==set(data['source_hashes'])
        for name,sha in data['source_hashes'].items():assert digest(z.read(name))==sha
        assert z.read('research/spatial-normal/plan.json')==(DIRECTORY/'plan.json').read_bytes()
        cmake=z.read('spatial_backend/CMakeLists.txt').decode();assert '2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5' in cmake and 'f7c486f10b9f645d0577249d02855e076e8368babaa3d31ced6095e7162baee9' in cmake
    authored=json.loads((directory/'scenes.json').read_text());expected={f'{c["id"]}/{mode}/{i}.json' for c in plan['scenes'] for mode in plan['modes'] for i in range(3)}
    with zipfile.ZipFile(directory/'traces.zip') as z:
        assert set(z.namelist())==expected
        traces={k:json.loads(z.read(k)) for k in expected}
    assert len(traces)==data['history_count']==36
    qualified_count=0
    for name,item in authored.items():
        scene=item['scene'];receipt=data['scenes'][name];U=np.asarray(scene['bodies'][0]['velocity']);N=len(scene['bodies'])-1
        passing={}
        for mode,settings in plan['modes'].items():
            costs=[];passed=True;saved=receipt['modes'][mode]
            for i in range(3):
                r=traces[f'{name}/{mode}/{i}.json'];s=np.asarray(r['states']);m=np.asarray(r['mass']);t=np.asarray(r['times'])
                assert s.shape==(5,N+1,13) and np.isfinite(s).all()
                np.testing.assert_allclose(t,[0,.01,.02,.03,.04],atol=1e-14,rtol=0)
                np.testing.assert_allclose(m,np.r_[0,np.ones(N)],atol=1e-13,rtol=1e-13)
                for j,body in enumerate(scene['bodies']):np.testing.assert_allclose(s[0,j,:3],body['position'],atol=1e-14,rtol=0)
                for j in range(1,N+1):np.testing.assert_allclose(r['inertia_body_kg_m2'][j],np.eye(3)*.004,atol=1e-14,rtol=0)
                for key,value in settings.items():assert r['numerical_model'][key]==value
                assert r['scalar_precision']=='float64'
                assert r['physical_setup_id']==digest(json.dumps(scene,sort_keys=True,separators=(',',':')).encode())
                exact=s[0,1:,:3]+t[:,None,None]*U
                diag=dict(max_position_error_m=float(np.max(np.sqrt(np.sum((s[:,1:,:3]-exact)**2,axis=2)))),max_velocity_error_m_s=float(np.max(np.sqrt(np.sum((s[1:,1:,7:10]-U)**2,axis=2)))),max_omega_rad_s=float(np.max(np.sqrt(np.sum(s[:,:,10:13]**2,axis=2)))),max_contact_penetration_m=r['max_contact_penetration_m'],max_closing_contact_speed_m_s=r['max_closing_contact_speed_m_s'],max_container_surface_excess_m=r.get('max_container_surface_excess_m',0),boundary_work_error_J=abs(r['boundary_work_J']-N*float(U@U)),energy_balance_error_J=abs(.5*np.sum(m[1:]*np.sum(s[-1,1:,7:10]**2,axis=1))-.5*N*float(U@U)),coupled_fallbacks=r['coupled_fallbacks'],normal_qp_rejections=r['normal_qp_rejections'])
                for key,v in diag.items():
                    # Independent energy summation differs at double roundoff,
                    # far below the unchanged 1e-5 J physical gate.
                    atol=1e-9 if key in ('energy_balance_error_J','boundary_work_error_J') else 1e-10
                    np.testing.assert_allclose(v,saved['diagnostics'][i][key],atol=atol,rtol=1e-8)
                passed &= all(diag[k]<=v for k,v in plan['analytic_gates'].items())
                assert r['normal_qp_solves']>0 and r['collision_updates']==sum(r['updates'])==r['coupled_updates'] and r['sequential_updates']==0
                assert r['mobility_matrix_bytes_max']==8*r['mobility_rows_max']**2
                assert r['mobility_rows_max']==saved['mobility_rows_max'] and r['mobility_matrix_bytes_max']==saved['mobility_matrix_bytes_max']
                assert r['normal_qp_solves']==saved['normal_qp_solves']
                if mode=='compact':assert r['eliminated_tangent_rows_max']==2*r['mobility_rows_max']
                if item['half'] is not None:
                    sampled=containment(scene,r,item['half']);assert r['max_container_surface_excess_m']>=sampled-1e-12
                np.testing.assert_allclose(s[:,0,:3],s[0,0,:3]+t[:,None]*U,atol=1e-11,rtol=0)
                np.testing.assert_allclose(np.linalg.norm(s[:,:,3:7],axis=2),1,atol=1e-13)
                reference=traces[f'{name}/{mode}/0.json'];assert r['states']==reference['states']
                assert digest(json.dumps(r['states'],sort_keys=True,separators=(',',':')).encode())==data['warmups'][f'{name}/{mode}']['states_sha256']
                assert r['step_s']>0;costs.append(r['step_s'])
            assert bool(passed)==saved['qualified'];passing[mode]=bool(passed)
            np.testing.assert_allclose(statistics.median(costs),saved['median_step_s'],atol=1e-12,rtol=1e-12)
        identity=all(traces[f'{name}/compact/{i}.json']['states']==traces[f'{name}/postassembly/{i}.json']['states'] for i in range(3))
        assert identity==receipt['bitwise_identical_states']
        qualified=identity and all(passing.values());assert qualified==receipt['qualified']
        if qualified:
            qualified_count+=1
            ratio=receipt['modes']['postassembly']['median_step_s']/receipt['modes']['compact']['median_step_s'];np.testing.assert_allclose(ratio,receipt['speedup'])
            ratio=receipt['modes']['postassembly']['mobility_matrix_bytes_max']/receipt['modes']['compact']['mobility_matrix_bytes_max'];assert ratio==receipt['mobility_payload_reduction']==9
        else:assert receipt['speedup'] is None and receipt['mobility_payload_reduction'] is None
    print(f'3D normal audit PASS: 36 histories, {qualified_count}/6 analytic gates, identical-state ablation and timing/matrix claims agree.')
    return data

if __name__=='__main__':audit()
