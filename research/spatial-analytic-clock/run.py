"""Fresh unchanged large-scene trajectories after component128 recovery."""
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import time
import sys
from spatial_engine import run
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'plan.json').read_text());D=H/'results'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
name=list(plan['cases'])[int(sys.argv[1])];D=H/'results'/name;D.mkdir(parents=True,exist_ok=False);entry=import_module('research.rapid-friction.run').scenes()[name]
assert hashlib.sha256(json.dumps(entry['scene'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==plan['cases'][name]
assert os.environ['OPENBLAS_NUM_THREADS']=='1' and os.environ['OMP_NUM_THREADS']=='1'
receipt=json.loads((H/'build-receipt.json').read_text());exe=Path(receipt['binary']);assert digest(exe)==receipt['binary_sha256'];assert json.loads((H/'controls/summary.json').read_text())['passed']
paths=[exe,H/'build-receipt.json',H/'controls/summary.json',*[Path(p) for p in receipt['transformed']],ROOT/'spatial_engine.py',H/'run.py',H/'plan.json',*[p for p in (ROOT/'spatial_backend').glob('*') if p.is_file()]];guards={str(p):digest(p) for p in paths}
import re
libs=re.findall(r'(/\S+)\s+\(',subprocess.check_output(['ldd',str(paths[0])],text=True))
save(D/'provenance.json',{'integrated_source':plan['source'],'execution_source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'guards':guards,'runtime':{str(Path(p).resolve()):digest(Path(p).resolve()) for p in libs}})
save(D/'scene.json',entry);records=[];refs={};edges=[]
for index,setting in enumerate(plan['settings']):
    T=D/f'trial_{index}';T.mkdir()
    start=time.perf_counter()
    try:
        result=run(entry['scene'],binary=exe,solver='coulomb',iterations=4096,kinematic_contact_phase='start',position_stabilization='split_translation_combined',contact_point_policy='shared',contact_tolerance_m_s=1e-8,contact_slop_m=1e-9,contact_recovery=True,early_component_recovery=True,rejected_contact_path=T/'rejection.json',progress_checkpoint_path=T/'progress.json',**setting)
        result['numerical_model']['analytic_clock_policy']=result['analytic_clock_policy'];base=import_module('research.rapid-friction.run');gates=json.loads((ROOT/'research/rapid-friction/plan.json').read_text());record={'complete':True,'result':result,'physical':base.physical(entry,result,gates),'setting':setting}
    except (RuntimeError,subprocess.CalledProcessError) as e:record={'complete':False,'error':str(e),'stderr':getattr(e,'stderr',None),'setting':setting}
    record['elapsed_s_descriptive']=time.perf_counter()-start;record['source_binary_unchanged']=all(digest(p)==sha for p,sha in guards.items());assert record['source_binary_unchanged'];save(T/'final.json',record);print({k:v for k,v in record.items() if k!='result'},flush=True)
    records.append({'trial':index,**{k:v for k,v in record.items() if k!='result'}});save(D/'summary.json',records)
    if record['complete']:
        refs[index]=record
        if index-1 in refs:
            error=base.errors(3,refs[index-1]['result'],record['result']);passed=refs[index-1]['physical']['passed'] and record['physical']['passed'] and all(error[k]<=v/4 for k,v in gates['trajectory_budgets']['3'].items());edges.append({'left':index-1,'right':index,'passed':bool(passed),'errors':error});save(D/'edges.json',edges);print('edge',name,index,passed,error,flush=True)
save(D/'final.json',{'complete':True,'source_binary_unchanged':True,'records':records})
