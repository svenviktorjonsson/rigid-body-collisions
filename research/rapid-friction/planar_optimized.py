"""Benchmark expanded numeric settings against sealed qualified planar roots."""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('rapid_frozen',HERE/'run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def main():
    plan=json.loads((HERE/'planar-optimized-plan.json').read_text());gates=json.loads((HERE/'plan.json').read_text());old=HERE/plan['reference_archive'];entries=json.loads((old/'scenes.json').read_text());previous=json.loads((old/'summary.json').read_text())
    out=HERE/'results-planar-optimized';out.mkdir(exist_ok=False);binary=ROOT/plan['binary'];budget=gates['trajectory_budgets']['2']
    paths=[binary,binary.parent/'precision-source.json',ROOT/'rigid_engine.py',HERE/'planar_optimized.py',HERE/'planar-optimized-plan.json',HERE/'run.py',old/'summary.json',old/'scenes.json']
    paths += list(old.glob('*/*reference*.json'))
    guards={str(p):base.digest(p) for p in paths}
    base.atomic(out/'scenes.json',entries);base.atomic(out/'provenance.json',{'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'guards':guards})
    def execute(name,entry,label,setting):
        dest=out/name/(label+'.json');assert not dest.exists();start=time.perf_counter()
        result=base.planar_run(entry['scene'],backend='block',binary=binary,**setting)
        physical=base.physical(entry,result,gates);physical['passed'] &= result['friction_impulse_abs_kg_m_s']>0
        record={'complete':True,'result':result,'physical':physical,'setting':setting,'process_elapsed_s':time.perf_counter()-start};base.atomic(dest,record);return record
    summary={}
    for name,entry in entries.items():
        assert previous[name]['reference_qualified'] and all(e['passed'] for e in previous[name]['phases'][0]['edges'][-2:])
        reference_record=json.loads((old/name/'tight_velocity128_reference_4.json').read_text());reference=reference_record['result'];passing=[];item={'reference_qualified':True,'reference_archive':plan['reference_archive'],'reference_record':name+'/tight_velocity128_reference_4.json','candidates':[]};summary[name]=item
        for primary in plan['primary_steps']:
            for iterations in plan['velocity_iterations']:
                setting={'dt':plan['dt'],'primary_steps':primary,'substeps':iterations,'position_iterations':plan['position_iterations']}
                record=execute(name,entry,f'candidate_{primary}_{iterations}',setting);error=base.errors(2,reference,record['result']);passed=record['physical']['passed'] and all(error[k]<=v for k,v in budget.items())
                candidate={'setting':setting,'passed':bool(passed),'errors':error,'native_s':record['result']['step_s']};item['candidates'].append(candidate)
                if passed:passing.append(candidate)
                print(name,primary,iterations,'passed',passed,'errors',error,flush=True)
        assert passing;selected=min(passing,key=lambda c:c['native_s']);item['selected']=selected
        variants={'reference':reference_record['setting'],'candidate':selected['setting']};samples={k:[] for k in variants};process_samples={k:[] for k in variants};states={k:[] for k in variants};ok=True
        def gate(record):
            error=base.errors(2,reference,record['result']);return record['physical']['passed'] and all(error[k]<=v for k,v in budget.items())
        for label,setting in variants.items():ok &= gate(execute(name,entry,'warmup_'+label,setting))
        for i in range(plan['timing_repetitions']):
            for label in (['reference','candidate'] if i%2==0 else ['candidate','reference']):
                record=execute(name,entry,f'timing_{i}_{label}',variants[label]);ok &= gate(record)
                samples[label].append(record['result']['step_s']);process_samples[label].append(record['result']['wall_time_s']);states[label].append(np.asarray(record['result']['states']).tobytes())
        deterministic=all(len(set(v))==1 for v in states.values());benchmark={'qualified':bool(ok and deterministic),'samples_s':samples,'adapter_process_samples_s':process_samples,'states_bitwise_repeated':deterministic,'settings':variants}
        if benchmark['qualified']:
            benchmark['median_s']={k:statistics.median(v) for k,v in samples.items()};benchmark['reference_over_candidate']=benchmark['median_s']['reference']/benchmark['median_s']['candidate']
            benchmark['adapter_process_median_s']={k:statistics.median(v) for k,v in process_samples.items()};benchmark['adapter_process_ratio']=benchmark['adapter_process_median_s']['reference']/benchmark['adapter_process_median_s']['candidate']
        item['benchmark']=benchmark;assert all(base.digest(p)==h for p,h in guards.items());base.atomic(out/'summary.json',summary);print(name,'BENCHMARK',benchmark,flush=True)
    base.atomic(out/'final.json',{'complete':True,'source_unchanged':True,'summary':summary})

if __name__=='__main__':main()
