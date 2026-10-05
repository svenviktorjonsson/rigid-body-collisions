"""Independent archived-state selection and provenance audit for shared geometry."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
import numpy as np
from research.audit_fast_shake_diagnostic import errors,physical
ROOT=Path(__file__).parents[1];D=ROOT/'research/shared-shake-study'
def sha(x):return hashlib.sha256(x).hexdigest()
def audit(directory=D,check_git=True):
    directory=Path(directory);out=directory/'results';s=json.loads((out/'summary.json').read_text());plan=json.loads((directory/'plan.json').read_text())
    assert s['plan_sha256']==sha((directory/'plan.json').read_bytes())
    for p,h in s['hashes'].items():assert sha((out/p).read_bytes())==h
    with zipfile.ZipFile(out/'execution-source.zip') as z:
        assert set(z.namelist())==set(s['source_hashes'])
        for p,h in s['source_hashes'].items():
            c=z.read(p);assert sha(c)==h
            if check_git:assert c==subprocess.check_output(['git','show',s['execution_source_commit']+':'+p],cwd=ROOT)
        assert z.read('research/shared-shake-study/plan.json')==(directory/'plan.json').read_bytes()
        scene=json.loads(z.read(plan['scene']))['scene']
    names=list(plan['reference_levels'])+plan['order'];runs={};physical_pass={}
    with zipfile.ZipFile(out/'traces.zip') as z:
        assert set(z.namelist())=={n+'.json' for n in names}
        for n in names:
            r=json.loads(z.read(n+'.json'));runs[n]=r
            if 'rejected' in r:physical_pass[n]=False;continue
            controls=plan['reference_levels'][n] if n in plan['reference_levels'] else plan['settings'][n.rsplit('_',1)[0]]
            states=np.asarray(r['states']);times=np.asarray(r['times']);assert states.shape==(round(scene['duration']/controls['dt'])+1,len(scene['bodies']),13) and np.isfinite(states).all()
            assert r['physical_setup_id']==sha(json.dumps(scene,sort_keys=True,separators=(',',':')).encode())
            np.testing.assert_allclose(states[0,:,:3],[b['position'] for b in scene['bodies']],rtol=0,atol=1e-14)
            np.testing.assert_allclose(times,np.arange(len(times))*controls['dt'],rtol=0,atol=1e-14)
            for k,v in plan['common'].items():
                if k=='contact_recovery':assert r['numerical_model'][k]['enabled']==v
                else:assert r['numerical_model'][k]==v
            for k in ('primary_steps','travel_fraction'):assert r['numerical_model'][k]==controls[k]
            assert r['collision_updates']==sum(r['updates']) and r['step_s']>0 and r['wall_time_s']>=r['step_s']
            p=physical(scene,r)
            for k,v in p.items():assert np.isclose(s['physical'][n][k],v,rtol=1e-9,atol=1e-10)
            physical_pass[n]=all(p[k]<=v for k,v in plan['physical_gates'].items()) and r['coulomb_residual_max_m_s']<=plan['common']['contact_tolerance_m_s'] and r['coupled_fallbacks']==0
    reference_good=all(physical_pass[n] for n in plan['reference_levels'])
    for a,b in plan['reference_edges']:
        e=errors(runs[a],runs[b]) if 'rejected' not in runs[a] and 'rejected' not in runs[b] else None
        good=e is not None and all(e[k]<=v for k,v in plan['reference_budget'].items());record=s['reference_edges'][a+'->'+b]
        assert record['qualified']==good
        if e is None:assert record['errors'] is None
        else:
            for k,v in e.items():assert np.isclose(record['errors'][k],v,rtol=1e-9,atol=1e-10)
        reference_good&=good
    assert s['reference_qualified']==reference_good
    finest=runs[list(plan['reference_levels'])[-1]];checks={}
    for n in names:
        if n in plan['reference_levels']:assert s['checks'][n]==physical_pass[n];continue
        if 'rejected' in runs[n]:assert s['checks'][n] is False;checks[n]=False;continue
        e=errors(runs[n],finest) if 'rejected' not in finest else None;good=physical_pass[n] and e is not None and all(e[k]<=v for k,v in plan['trajectory_budget'].items())
        record=s['checks'][n];assert record['physical_pass']==physical_pass[n] and record['qualified']==good
        if e is None:assert record['errors'] is None
        else:
            for k,v in e.items():assert np.isclose(record['errors'][k],v,rtol=1e-9,atol=1e-10)
        checks[n]=good
    lanes_good=[]
    for lane in plan['settings']:
        names=[n for n in plan['order'] if n.rsplit('_',1)[0]==lane];good=all(checks[n] for n in names)
        identical=good and all(runs[n]['states']==runs[names[0]]['states'] for n in names);record=s['lanes'][lane]
        assert record['qualified']==(good and identical) and record['bitwise_repeatable']==identical
        native=[runs[n]['step_s'] for n in names if 'rejected' not in runs[n]];whole=[runs[n]['wall_time_s'] for n in names if 'rejected' not in runs[n]]
        assert record['native_s']==native and record['whole_s']==whole
        assert record['median_native_s']==(float(np.median(native)) if native else None);lanes_good.append(good and identical)
    assert s['qualified']==bool(reference_good and all(lanes_good))
    assert s['median_native_ratio']==(s['lanes']['reference']['median_native_s']/s['lanes']['candidate']['median_native_s'] if s['qualified'] else None)
    print('Shared shake audit PASS:',len(runs),'attempts; references',s['reference_qualified'],'qualified',s['qualified'],'ratio',s['median_native_ratio']);return s
if __name__=='__main__':audit()
