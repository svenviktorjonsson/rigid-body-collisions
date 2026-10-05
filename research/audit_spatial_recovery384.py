"""Paired source/geometry audit of the retained gyroscopic-RHS follow-up."""
import hashlib
import json
from pathlib import Path
import zipfile
DIRECTORY=Path(__file__).parent/'spatial-friction-recovery384'
def sha(b):return hashlib.sha256(b).hexdigest()

def audit():
    directory=DIRECTORY/'results';s=json.loads((directory/'summary.json').read_text());plan=json.loads((DIRECTORY/'plan.json').read_text())
    assert s['execution_source_commit']=='4910fbca7cacf8dfd218be97a082d32b7ea13ea2'
    assert s['plan_sha256']==sha((DIRECTORY/'plan.json').read_bytes())
    for name,digest in s['hashes'].items():assert sha((directory/name).read_bytes())==digest
    with zipfile.ZipFile(directory/'execution-source.zip') as z:
        assert set(z.namelist())==set(s['source_hashes'])
        for name,digest in s['source_hashes'].items():assert sha(z.read(name))==digest
        assert z.read('research/spatial-friction-recovery384/plan.json')==(DIRECTORY/'plan.json').read_bytes()
        assert 'consistentTangentRHS(m_b[i]' in z.read('spatial_backend/coulomb.h').decode()
        assert 'circular_polish::solve' in z.read('spatial_backend/coulomb.h').decode()
        assert 'remaining_svd_calls=256' in z.read('spatial_backend/coulomb_polish.h').decode()
        assert 'if(n>384)' in z.read('spatial_backend/coulomb_polish.h').decode()
        assert 'column_correlation<=1e-10' in z.read('spatial_backend/newton_linear.h').decode()
        assert 'GyroInspect gyro;gyro.check()' in z.read('spatial_backend/friction_checks.cpp').decode()
    baseline=Path(__file__).parent/'spatial-friction-recovery'
    oldplan=json.loads((baseline/'plan.json').read_text());oldscenes=json.loads((baseline/'results/scenes.json').read_text());scenes=json.loads((directory/'scenes.json').read_text())
    assert plan['baseline_source_commit']=='f6d03878e7d187b18be8d590c6e66f976232e020'
    assert plan['common']==dict(oldplan['common'],contact_recovery=True)
    for key in ['dt_s','trajectory_budget','physical_gates']:assert plan[key]==oldplan[key]
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
    assert s['attempt_count']==3 and s['history_count']==0
    print('Native recovery follow-up audit PASS: exact paired physics, bounded same-law recovery, 3 retained rejections; no hull trajectory qualification.')
    return s
if __name__=='__main__':audit()
