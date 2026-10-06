"""Prospective unchanged rotating-scene joint normal/friction reference ladders."""
import hashlib,json,os,subprocess,time,re
from importlib import import_module
from pathlib import Path
import numpy as np
from rigid_engine import run
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'plan.json').read_text());D=H/'world-results';D.mkdir(exist_ok=False)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((H/'controls-v2/summary.json').read_text())['passed']
assert os.environ['OMP_NUM_THREADS']=='1' and os.environ['OPENBLAS_NUM_THREADS']=='1'
exe=ROOT/plan['binary'];receipt=json.loads((H/'build-receipt.json').read_text());assert sha(exe)==receipt['binary_sha256']
paths=[Path(__file__),H/'plan.json',H/'build-receipt.json',H/'controls-v2/summary.json',exe,exe.parent/'precision-source.json',ROOT/'rigid_engine.py',ROOT/'research/rapid-friction/run.py'];guards={str(p):sha(p) for p in paths}
libs=re.findall(r'(/\S+)\s+\(',subprocess.check_output(['ldd',str(exe)],text=True));save(D/'provenance.json',{'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'guards':guards,'runtime':{str(Path(p).resolve()):sha(Path(p).resolve()) for p in libs}})
base=import_module('research.rapid-friction.run');gates=json.loads((ROOT/'research/rapid-friction/plan.json').read_text());scenes=base.scenes();save(D/'scenes.json',{k:scenes[k] for k in plan['cases']});summary={}
for name in plan['cases']:
 entry=scenes[name];target=D/name;target.mkdir();refs=[];edges=[]
 for i,setting in enumerate(plan['reference_levels']):
  start=time.perf_counter();result=run(entry['scene'],backend='block',binary=exe,substeps=plan['velocity_iterations'],position_iterations=plan['position_iterations'],**setting)
  result['numerical_model'].update(joint_normal_tangent_block_solver=True,joint_root_tolerance_m_s=1e-10,joint_max_faces=16,joint_max_unknowns=4,joint_linear_rank_relative_cutoff=1e-12,joint_fallback='unchanged_original_contact_iteration_if_no_checked_root')
  if i==0:
   old=json.loads((ROOT/'research/rapid-friction/results-planar-discovery'/name/'seams_off_reference_0.json').read_text())['result']
   parity={'mass_max_error':float(np.max(abs(np.asarray(old['mass'])-result['mass']))),'inertia_max_error':float(np.max(abs(np.asarray(old['inertia'])-result['inertia']))),'initial_state_max_error':float(np.max(abs(np.asarray(old['states'][0])-result['states'][0])))}
   save(target/'authored-model-parity.json',parity);assert all(v<=1e-10 for v in parity.values()),parity
  record={'complete':True,'result':result,'physical':base.physical(entry,result,gates),'setting':setting,'elapsed_s_descriptive':time.perf_counter()-start}
  save(target/f'reference_{i}.json',record);refs.append(record)
  if i:
   error=base.errors(2,refs[-2]['result'],result);passed=refs[-2]['physical']['passed'] and record['physical']['passed'] and all(error[k]<=v/4 for k,v in gates['trajectory_budgets']['2'].items());edges.append({'left':i-1,'right':i,'passed':bool(passed),'errors':error});print(name,i,'edge',passed,error,flush=True)
  summary[name]={'full_histories':len(refs),'physical_passes':sum(r['physical']['passed'] for r in refs),'edges':edges,'reference_qualified':len(edges)>=2 and all(e['passed'] for e in edges[-2:]),'performance_qualified':False};save(D/'summary.json',summary)
  assert all(sha(p)==v for p,v in guards.items())
save(D/'final.json',{'complete':True,'source_binary_unchanged':True,'summary':summary})
