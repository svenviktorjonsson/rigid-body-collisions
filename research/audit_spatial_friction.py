"""Independently recompute archived circular-friction trajectory qualifications."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from spatial_engine import errors
from research.spatial_metrics import diagnostics
DIRECTORY=Path(__file__).parent/'spatial-friction'
def digest(data):return hashlib.sha256(data).hexdigest()

def audit():
    directory=DIRECTORY/'results';summary=json.loads((directory/'summary.json').read_text());plan=json.loads((DIRECTORY/'plan.json').read_text())
    assert summary['execution_source_commit']=='7c279768f6518e66b1ee620d44bd9b37309b1b5f'
    assert summary['plan_sha256']==digest((DIRECTORY/'plan.json').read_bytes())
    for name,sha in summary['hashes'].items():assert digest((directory/name).read_bytes())==sha
    with zipfile.ZipFile(directory/'execution-source.zip') as z:
        assert set(z.namelist())==set(summary['source_hashes'])
        for name,sha in summary['source_hashes'].items():assert digest(z.read(name))==sha
        assert z.read('research/spatial-friction/plan.json')==(DIRECTORY/'plan.json').read_bytes()
        cmake=z.read('spatial_backend/CMakeLists.txt').decode();assert '2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5' in cmake
        assert 'setCenterOfMassTransform' in z.read('spatial_backend/runner.cpp').decode()
    authored=json.loads((directory/'scenes.json').read_text())
    expected={f'{c["id"]}/{name}.json' for c in plan['scenes'] for name in
              [*[f'reference_{i}' for i in range(len(c['fractions']))],*[f'{lane}_{i}' for lane in plan['candidates'] for i in range(plan['candidate_repetitions'])]]}
    with zipfile.ZipFile(directory/'traces.zip') as z:
        assert set(z.namelist())==expected;runs={name:json.loads(z.read(name)) for name in expected}
    histories=0;references=0;recommendations=0
    for config in plan['scenes']:
        name=config['id'];scene=authored[name]['scene'];half=authored[name]['half'];record=summary['scenes'][name]
        states={};physical={};diagnostic={}
        lanes={f'reference_{i}':fraction for i,fraction in enumerate(config['fractions'])}
        lanes.update({f'{candidate}_{i}':fraction for candidate,fraction in plan['candidates'].items() for i in range(plan['candidate_repetitions'])})
        for lane,fraction in lanes.items():
            r=runs[f'{name}/{lane}.json'];states[lane]=r
            if 'rejected' in r:
                assert r['exit_code']==1 and r['elapsed_s']>0 and r['rejected'];physical[lane]=False;continue
            histories+=1;s=np.asarray(r['states']);assert s.shape==(13,len(scene['bodies']),13) and np.isfinite(s).all()
            assert r['physical_setup_id']==digest(json.dumps(scene,sort_keys=True,separators=(',',':')).encode())
            np.testing.assert_allclose(r['times'],np.arange(13)*.01,atol=1e-14,rtol=0)
            for i,body in enumerate(scene['bodies']):np.testing.assert_allclose(s[0,i,:3],body['position'],atol=1e-14,rtol=0)
            for key,value in plan['common'].items():assert r['numerical_model'][key]==value
            assert r['numerical_model']['travel_fraction']==fraction
            assert r['scalar_precision']=='float64' and r['coupled_fallbacks']==0 and r['sequential_updates']==0
            assert r['coulomb_residual_max_m_s']<=plan['common']['contact_tolerance_m_s']
            assert r['coulomb_sweeps_max']<=plan['common']['iterations'] and 0<=r['coulomb_fast_solves']<=r['coulomb_solves']
            assert r['collision_updates']==sum(r['updates'])==r['coupled_updates']
            assert r['mobility_matrix_bytes_max']==8*r['mobility_rows_max']**2 and r['step_s']>0
            d=diagnostics(scene,r,half);diagnostic[lane]=d
            physical[lane]=all(d[key]<=limit for key,limit in plan['physical_gates'].items())
            for key,value in d.items():np.testing.assert_allclose(value,record['diagnostics'][lane][key],atol=1e-10,rtol=1e-9)
        assert set(record['diagnostics'])==set(diagnostic)
        edges=[]
        for i in range(len(config['fractions'])-1):
            left=f'reference_{i}';right=f'reference_{i+1}'
            error=errors(states[left],states[right]) if physical[left] and physical[right] else None
            passed=bool(error and all(error[key]<=limit/4 for key,limit in plan['trajectory_budget'].items()))
            edge=record['edges'][i];assert edge['left']==left and edge['right']==right and edge['passed']==passed
            if error:
                for key,value in error.items():np.testing.assert_allclose(value,edge['error'][key],atol=1e-12,rtol=1e-10)
            else:assert edge['error'] is None
            edges.append(passed)
        qualified=all(edges[-2:]);assert qualified==record['reference_qualified'];references+=qualified
        reference=states[f'reference_{len(config["fractions"])-1}'];passing=[];costs={}
        for candidate in plan['candidates']:
            names=[f'{candidate}_{i}' for i in range(plan['candidate_repetitions'])];good=True;cost=[]
            for i,lane in enumerate(names):
                error=errors(reference,states[lane]) if qualified and physical[lane] else None
                good &= bool(error and all(error[key]<=limit for key,limit in plan['trajectory_budget'].items()))
                saved=record['candidates'][candidate]['errors'][i]
                if error:
                    for key,value in error.items():np.testing.assert_allclose(value,saved[key],atol=1e-12,rtol=1e-10)
                else:assert saved is None
                if 'rejected' not in states[lane]:cost.append(states[lane]['step_s'])
            identity=len(cost)==len(names) and all(states[lane]['states']==states[names[0]]['states'] for lane in names)
            saved=record['candidates'][candidate];assert identity==saved['deterministic_states'] and bool(good and identity)==saved['qualified']
            if cost:np.testing.assert_allclose(np.median(cost),saved['median_step_s'],atol=1e-12,rtol=1e-12)
            else:assert saved['median_step_s'] is None
            fast=[states[lane]['coulomb_fast_solves']/max(1,states[lane]['coulomb_solves']) for lane in names if 'rejected' not in states[lane]]
            np.testing.assert_allclose(fast,saved['native_fast_solve_fraction'],atol=1e-14,rtol=0)
            if good and identity:passing.append(candidate);costs[candidate]=float(np.median(cost))
        choice=min(passing,key=costs.get) if passing else None;assert choice==record['choice'];recommendations+=choice is not None
    assert len(runs)==summary['attempt_count'] and histories==summary['history_count']
    print(f'3D friction audit PASS: {len(runs)} attempts, {histories} histories, {references}/6 reference gates, {recommendations} qualified choices; all rejections retained.')
    return summary
if __name__=='__main__':audit()
