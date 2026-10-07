"""Declared original rotating cases with tighter slop and seam controls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from rigid_engine import run
from importlib import import_module
base=import_module('research.rapid-friction.run')
H=Path(__file__).resolve().parent
ROOT=H.parents[1]
plan=json.loads((H/'world-plan.json').read_text())
D=H/'world-results';D.mkdir(exist_ok=False)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
assert os.environ['OMP_NUM_THREADS']=='1' and os.environ['OPENBLAS_NUM_THREADS']=='1'
paths=[H/'world.py',H/'world-plan.json',ROOT/'research/rapid-friction/run.py',ROOT/'rigid_engine.py',ROOT/plan['binary'],ROOT/plan['binary'].replace('rigid_runner','precision-source.json'),H/'build-receipt.json',H/'runner.cpp']
guards={str(p):digest(p) for p in paths}
save(D/'provenance.json',{'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'guards':guards,'scope':'numerical discovery/slop experiment, no speed acceptance'})
scenes=base.scenes();gates=json.loads((ROOT/'research/rapid-friction/plan.json').read_text());authored={};summary={}
for name in plan['cases']:
    entry=scenes[name];item={'reference_qualified':False,'phases':[],'benchmark':None};summary[name]=item
    for suppress in plan['phases'][name]:
        entry=copy.deepcopy(scenes[name]);entry['scene']['suppress_internal_edges']=suppress
        phase_id='seams_on' if suppress else 'seams_off';authored[name]=entry
        save(D/'scenes.json',authored)
        refs=[];edges=[]
        for i,setting in enumerate(plan['reference_levels']):
            start=time.perf_counter();result=run(entry['scene'],backend='block',binary=ROOT/plan['binary'],substeps=plan['substeps'],position_iterations=plan['position_iterations'],**setting)
            record={'complete':True,'result':result,'setting':setting,'physical':base.physical(entry,result,gates),'process_elapsed_s':time.perf_counter()-start,'suppress_internal_edges':suppress}
            save(D/name/f'{phase_id}_reference_{i}.json',record);refs.append(record)
            if not suppress:
                import numpy as np
                old=json.loads((ROOT/'research/rapid-friction/results-planar-discovery'/name/f'{phase_id}_reference_{i}.json').read_text())
                assert np.array(old['result']['states'],dtype=np.float64).tobytes()==np.array(result['states'],dtype=np.float64).tobytes()
                record['original_default_states_exact']=True
                save(D/name/f'{phase_id}_reference_{i}.json',record)
            if i:
                a,b=refs[-2:];error=base.errors(2,a['result'],b['result']);passed=a['physical']['passed'] and b['physical']['passed'] and all(error[k]<=v/4 for k,v in gates['trajectory_budgets']['2'].items())
                edges.append({'left':i-1,'right':i,'passed':bool(passed),'errors':error})
                print(name,phase_id,i,'edge',passed,error,flush=True)
        qualified=all(e['passed'] for e in edges[-2:]);item['phases'].append({'phase':{'id':phase_id,'suppress_internal_edges':suppress},'edges':edges,'qualified':qualified});item['reference_qualified'] |= qualified
        save(D/'summary.json',summary)
assert all(digest(p)==sha for p,sha in guards.items())
save(D/'final.json',{'complete':True,'source_unchanged':True,'summary':summary})
