"""Independent retained-history/physics/selection audit, not a timing assertion."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from research.audit_fast_shake_diagnostic import audit as audit_references,errors,physical
ROOT=Path(__file__).parents[1];DIRECTORY=ROOT/'research/shake-performance'
def sha(value):return hashlib.sha256(value).hexdigest()
def audit(directory=DIRECTORY,check_git=True):
    assert all(audit_references().values())
    directory=Path(directory);r=directory/'results';s=json.loads((r/'summary.json').read_text());plan=json.loads((directory/'plan.json').read_text())
    assert s['plan_sha256']==sha((directory/'plan.json').read_bytes())
    for name,digest in s['hashes'].items():assert sha((r/name).read_bytes())==digest
    with zipfile.ZipFile(r/'execution-source.zip') as z:
        assert set(z.namelist())==set(s['source_hashes'])
        for name,digest in s['source_hashes'].items():
            content=z.read(name);assert sha(content)==digest
            if check_git:assert content==subprocess.check_output(['git','show',s['execution_source_commit']+':'+name],cwd=ROOT)
        assert z.read('research/shake-performance/plan.json')==(directory/'plan.json').read_bytes()
        scene=json.loads(z.read(plan['scene']))['scene']
        refs={key:json.loads(z.read('research/fast-shake-diagnostic/'+file)) for key,file in [('fixed','fixed_1_25us_corrected.json'),('guard','eighth_travel.json')]}
    canonical=json.dumps(scene,sort_keys=True,separators=(',',':')).encode();checks={};runs={}
    with zipfile.ZipFile(r/'traces.zip') as z:
        assert set(z.namelist())=={name+'.json' for name in plan['order']}
        for name in plan['order']:
            result=json.loads(z.read(name+'.json'));runs[name]=result;lane=name.rsplit('_',1)[0]
            if 'rejected' in result:checks[name]=False;assert not s['checks'][name]['qualified'];continue
            controls=plan['settings'][lane];state=np.asarray(result['states']);times=np.asarray(result['times'])
            assert state.shape==(round(scene['duration']/controls['dt'])+1,28,13) and np.isfinite(state).all()
            assert result['physical_setup_id']==sha(canonical)
            np.testing.assert_allclose(state[0,:,:3],[b['position'] for b in scene['bodies']],rtol=0,atol=1e-14)
            np.testing.assert_allclose(times,np.arange(len(times))*controls['dt'],rtol=0,atol=1e-14)
            for key,value in plan['common'].items():
                if key!='contact_recovery':assert result['numerical_model'][key]==value
            assert result['numerical_model']['contact_recovery']['enabled']==plan['common']['contact_recovery']
            for key in ('primary_steps','travel_fraction'):assert result['numerical_model'][key]==controls[key]
            assert result['coulomb_residual_max_m_s']<=plan['common']['contact_tolerance_m_s'] and result['coupled_fallbacks']==0
            assert result['collision_updates']==sum(result['updates']) and result['step_s']>0 and result['wall_time_s']>=result['step_s']
            d=physical(scene,result);e={key:errors(result,reference) for key,reference in refs.items()}
            good=all(d[k]<=v for k,v in plan['physical_gates'].items()) and all(error[k]<=v for error in e.values() for k,v in plan['trajectory_budget'].items())
            record=s['checks'][name]
            for key,value in d.items():assert np.isclose(record['physical'][key],value,rtol=1e-9,atol=1e-10)
            for ref,error in e.items():
                for key,value in error.items():assert np.isclose(record['errors'][ref][key],value,rtol=1e-9,atol=1e-10)
            assert record['qualified']==good;checks[name]=good
    eligible=[]
    for lane in plan['settings']:
        names=[name for name in plan['order'] if name.rsplit('_',1)[0]==lane];record=s['lanes'][lane]
        good=all(checks[name] for name in names);identical=good and all(runs[name]['states']==runs[names[0]]['states'] for name in names)
        assert record['qualified']==(good and identical) and record['bitwise_repeatable']==identical
        native=[runs[name]['step_s'] for name in names if 'rejected' not in runs[name]];whole=[runs[name]['wall_time_s'] for name in names if 'rejected' not in runs[name]]
        assert record['native_s']==native and record['whole_process_s']==whole
        assert record['updates']==[runs[name]['collision_updates'] for name in names if 'rejected' not in runs[name]]
        if native:assert record['median_native_s']==float(np.median(native)) and record['median_whole_process_s']==float(np.median(whole))
        eligible.append(good and identical)
    assert s['qualified']==all(eligible)
    if s['qualified']:assert s['median_native_ratio']==s['lanes']['reference']['median_native_s']/s['lanes']['candidate']['median_native_s']
    else:assert s['median_native_ratio'] is None
    print('Repeated shake cost audit PASS:',len(runs),'retained attempts; qualified',s['qualified'],'native median ratio',s['median_native_ratio'])
    return s
if __name__=='__main__':audit()
