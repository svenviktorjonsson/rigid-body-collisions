"""Fresh unchanged large-scene trajectories after component128 recovery."""
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import time
from spatial_engine import run
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'plan.json').read_text());D=H/'results';D.mkdir(exist_ok=False)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
entry=json.loads((ROOT/'research/rapid-friction/results-large-irregular/scenes.json').read_text())[plan['case']]
assert hashlib.sha256(json.dumps(entry['scene'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==plan['scene_sha256']
assert os.environ['OPENBLAS_NUM_THREADS']=='1' and os.environ['OMP_NUM_THREADS']=='1'
paths=[*[ROOT/a['path'] for a in plan['anchors']],ROOT/'build/spatial/spatial_runner',ROOT/'spatial_engine.py',H/'run.py',H/'plan.json',*[p for p in (ROOT/'spatial_backend').glob('*') if p.is_file()]];guards={str(p):digest(p) for p in paths}
import re
libs=re.findall(r'(/\S+)\s+\(',subprocess.check_output(['ldd',str(paths[0])],text=True))
save(D/'provenance.json',{'integrated_source':plan['source'],'execution_source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'guards':guards,'runtime':{str(Path(p).resolve()):digest(Path(p).resolve()) for p in libs}})
save(D/'scene.json',entry);records=[]
for index,setting in enumerate(plan['settings']):
    T=D/f'trial_{index}';T.mkdir()
    start=time.perf_counter()
    if index<2:
        anchor=plan['anchors'][index];assert digest(ROOT/anchor['path'])==anchor['sha256'];record=json.loads((ROOT/anchor['path']).read_text());assert record['complete'] and record['physical']['passed'] and record['setting']==setting;save(T/'final.json',record);records.append({'trial':index,'authenticated_anchor':anchor,**{k:v for k,v in record.items() if k!='result'}});save(D/'summary.json',records);continue
    try:
        result=run(entry['scene'],solver='coulomb',iterations=4096,kinematic_contact_phase='start',position_stabilization='split_translation_combined',contact_point_policy='shared',contact_tolerance_m_s=plan['numerical_search_tolerance_m_s'],contact_slop_m=plan['contact_slop_m'],contact_recovery=True,early_component_recovery=True,rejected_contact_path=T/'rejection.json',progress_checkpoint_path=T/'progress.json',**setting)
        base=import_module('research.rapid-friction.run');gates=json.loads((ROOT/'research/rapid-friction/plan.json').read_text());record={'complete':True,'result':result,'physical':base.physical(entry,result,gates),'setting':setting}
    except (RuntimeError,subprocess.CalledProcessError) as e:record={'complete':False,'error':str(e),'stderr':getattr(e,'stderr',None),'setting':setting}
    record['elapsed_s_descriptive']=time.perf_counter()-start;record['source_binary_unchanged']=all(digest(p)==sha for p,sha in guards.items());assert record['source_binary_unchanged'];save(T/'final.json',record);print({k:v for k,v in record.items() if k!='result'},flush=True)
    records.append({'trial':index,**{k:v for k,v in record.items() if k!='result'}});save(D/'summary.json',records)
save(D/'final.json',{'complete':True,'source_binary_unchanged':True,'records':records})
