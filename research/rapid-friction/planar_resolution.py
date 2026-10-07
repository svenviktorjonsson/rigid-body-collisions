"""Higher iteration qualification under unchanged planar scene/material/gates."""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rapid_frozen',HERE/'run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def main():
    plan=json.loads((HERE/'planar-resolution-plan.json').read_text());gates=json.loads((HERE/'plan.json').read_text())
    original=json.loads((HERE/'results/scenes.json').read_text())
    entries={k:v for k,v in original.items() if v['dimension']==2}
    out=HERE/'results-planar-resolution';out.mkdir(exist_ok=False)
    assert os.environ.get('OPENBLAS_NUM_THREADS')=='1'
    guards={str(p):base.digest(p) for p in [HERE/'planar_resolution.py',HERE/'planar-resolution-plan.json',HERE/'plan.json',HERE/'run.py',base.ROOT/'build/rigid_double_ledger/rigid_runner',base.ROOT/'rigid_backend/runner.cpp',base.ROOT/'rigid_backend/compat2.h']}
    base.atomic(out/'provenance.json',{'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'guards':guards,'scene_source_sha256':base.digest(HERE/'results/scenes.json')})
    def execute(name,entry,label,setting):
        dest=out/name/(label+'.json');assert not dest.exists()
        begun=time.perf_counter()
        try:
            result=base.planar_run(entry['scene'],backend='block',binary=base.ROOT/'build/rigid_double_ledger/rigid_runner',**setting)
            record={'complete':True,'result':result,'physical':base.physical(entry,result,gates),'setting':setting,'process_elapsed_s':time.perf_counter()-begun}
        except RuntimeError as e:record={'complete':False,'error':str(e),'setting':setting,'process_elapsed_s':time.perf_counter()-begun}
        base.atomic(dest,record);return record
    summary={};budget=gates['trajectory_budgets']['2']
    for name,entry in entries.items():
        item={'phases':[],'reference_qualified':False,'benchmark':None};summary[name]=item
        for phase in plan['phases']:
            refs=[];edges=[];settings=[]
            for i,level in enumerate(plan['reference_levels']):
                setting=dict(level,substeps=phase['substeps'],position_iterations=phase['position_iterations']);settings.append(setting)
                record=execute(name,entry,phase['id']+f'_reference_{i}',setting);refs.append(record)
                if i:
                    a,b=refs[-2:];error=base.errors(2,a['result'],b['result']) if a['complete'] and b['complete'] else None
                    passed=bool(error is not None and a['physical']['passed'] and b['physical']['passed'] and all(error[k]<=v/4 for k,v in budget.items()))
                    edges.append({'passed':passed,'errors':error});print(name,phase['id'],'EDGE',i,edges[-1],flush=True)
            qualified=all(e['passed'] for e in edges);item['phases'].append({'phase':phase,'edges':edges,'qualified':qualified})
            base.atomic(out/'summary.json',summary)
            if not qualified:continue
            item['reference_qualified']=True;reference=refs[-1]['result'];passing=[];item['candidates']=[]
            for primary in plan['candidate_primary_steps']:
                for iterations in plan['candidate_substeps']:
                    setting={'dt':.01,'primary_steps':primary,'substeps':iterations,'position_iterations':phase['position_iterations']}
                    label=f'candidate_{primary}_{iterations}';record=execute(name,entry,label,setting)
                    error=base.errors(2,reference,record['result']) if record['complete'] else None
                    passed=bool(record['complete'] and record['physical']['passed'] and all(error[k]<=v for k,v in budget.items()))
                    candidate={'setting':setting,'passed':passed,'errors':error,'native_s':record['result']['step_s'] if record['complete'] else None};item['candidates'].append(candidate)
                    if passed:passing.append(candidate)
            passing.append({'setting':settings[0],'passed':True,'errors':base.errors(2,reference,refs[0]['result']),'native_s':refs[0]['result']['step_s']})
            assert all(passing[-1]['errors'][k]<=v for k,v in budget.items())
            selected=min(passing,key=lambda r:r['native_s']);item['selected']=selected
            variants={'reference':settings[-1],'candidate':selected['setting']};samples={k:[] for k in variants};states={k:[] for k in variants};ok=True
            for label,setting in variants.items():
                record=execute(name,entry,'warmup_'+label,setting);ok &= record['complete'] and record['physical']['passed']
            for repetition in range(plan['timing_repetitions']):
                for label in (['reference','candidate'] if repetition%2==0 else ['candidate','reference']):
                    record=execute(name,entry,f'timing_{repetition}_{label}',variants[label]);passed=record['complete'] and record['physical']['passed']
                    if record['complete']:
                        error=base.errors(2,reference,record['result']);passed &= all(error[k]<=v for k,v in budget.items())
                        samples[label].append(record['result']['step_s']);states[label].append(np.asarray(record['result']['states']).tobytes())
                    ok &= passed
            deterministic=all(len(set(x))==1 for x in states.values());benchmark={'qualified':bool(ok and deterministic),'samples_s':samples,'states_bitwise_repeated':deterministic,'settings':variants}
            if benchmark['qualified']:
                benchmark['median_s']={k:statistics.median(x) for k,x in samples.items()};benchmark['reference_over_candidate']=benchmark['median_s']['reference']/benchmark['median_s']['candidate']
            item['benchmark']=benchmark;break
        assert all(base.digest(p)==h for p,h in guards.items())
        base.atomic(out/'summary.json',summary);print(name,'RESULT',item['reference_qualified'],item['benchmark'],flush=True)
    base.atomic(out/'final.json',{'complete':True,'guards_unchanged':True,'summary':summary})

if __name__=='__main__':main()
