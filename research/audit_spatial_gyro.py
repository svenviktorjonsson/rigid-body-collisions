"""Paired source/geometry audit of the retained gyroscopic-RHS follow-up."""
import hashlib
import json
from pathlib import Path
import zipfile
DIRECTORY=Path(__file__).parent/'spatial-friction-gyro'
def sha(b):return hashlib.sha256(b).hexdigest()

def audit():
    directory=DIRECTORY/'results';s=json.loads((directory/'summary.json').read_text());plan=json.loads((DIRECTORY/'plan.json').read_text())
    assert s['execution_source_commit']=='78e63313b5d13dd80170d03c54bcaffccbbf47e8'
    assert s['plan_sha256']==sha((DIRECTORY/'plan.json').read_bytes())
    for name,digest in s['hashes'].items():assert sha((directory/name).read_bytes())==digest
    with zipfile.ZipFile(directory/'execution-source.zip') as z:
        assert set(z.namelist())==set(s['source_hashes'])
        for name,digest in s['source_hashes'].items():assert sha(z.read(name))==digest
        assert z.read('research/spatial-friction-gyro/plan.json')==(DIRECTORY/'plan.json').read_bytes()
        assert 'consistentTangentRHS(m_b[i]' in z.read('spatial_backend/coulomb.h').decode()
        assert 'GyroInspect gyro;gyro.check()' in z.read('spatial_backend/friction_checks.cpp').decode()
    baseline=Path(__file__).parent/'spatial-friction'
    oldplan=json.loads((baseline/'plan.json').read_text());oldscenes=json.loads((baseline/'results/scenes.json').read_text());scenes=json.loads((directory/'scenes.json').read_text())
    assert plan['baseline_source_commit']=='7c279768f6518e66b1ee620d44bd9b37309b1b5f'
    for key in ['common','dt_s','trajectory_budget','physical_gates']:assert plan[key]==oldplan[key]
    expected={f'{c["id"]}/reference_{i}.json' for c in plan['scenes'] for i in range(len(c['fractions']))}
    with zipfile.ZipFile(directory/'traces.zip') as z:
        assert set(z.namelist())==expected
        for name in expected:
            r=json.loads(z.read(name));assert r['exit_code']==1 and r['elapsed_s']>0
            assert r['rejected'].startswith('Coulomb residual gate failed') and 'no friction-law fallback' in r['rejected']
    for c in plan['scenes']:
        name=c['id'];assert c==next(x for x in oldplan['scenes'] if x['id']==name)
        assert scenes[name]==oldscenes[name]
        receipt=s['scenes'][name]
        assert receipt['reference_qualified'] is False and receipt['choice'] is None
        assert receipt['candidates']=={} and receipt['diagnostics']=={}
        assert all(e['passed'] is False and e['error'] is None for e in receipt['edges'])
    assert s['attempt_count']==6 and s['history_count']==0
    print('Gyroscopic follow-up audit PASS: exact paired physics/controls, 6 retained rejections; no hull trajectory qualification.')
    return s
if __name__=='__main__':audit()
