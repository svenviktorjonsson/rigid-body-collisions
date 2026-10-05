"""Predeclared repeated whole-trajectory cost comparison, retaining failures."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import platform
import time
import zipfile
import numpy as np
from spatial_engine import run,BINARY
from research.audit_fast_shake_diagnostic import audit as audit_references,errors,physical
ROOT=Path(__file__).parents[1];DIRECTORY=ROOT/'research/shake-performance'
SOURCES=['spatial_engine.py','spatial_backend/runner.cpp','spatial_backend/coulomb.h',
 'spatial_backend/coulomb_polish.h','spatial_backend/newton_linear.h','spatial_backend/normal_qp.h',
 'spatial_backend/CMakeLists.txt','research/run_shake_performance.py','research/shake-performance/plan.json',
 'research/audit_fast_shake_diagnostic.py','research/fast-shake-diagnostic/scene.json',
 'research/fast-shake-diagnostic/fixed_1_25us_corrected.json','research/fast-shake-diagnostic/eighth_travel.json']
def sha(value):return hashlib.sha256(value).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def main():
    assert all(audit_references().values())
    plan=json.loads((DIRECTORY/'plan.json').read_text());source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in SOURCES:
        if subprocess.check_output(['git','show',source+':'+path],cwd=ROOT)!=(ROOT/path).read_bytes():raise RuntimeError('Freeze before execution: '+path)
    results=DIRECTORY/'results';results.mkdir(exist_ok=True);checkpoint=results/'checkpoints';checkpoint.mkdir(exist_ok=True)
    provenance=dict(execution_source_commit=source,binary_sha256=sha(BINARY.read_bytes()),plan_sha256=sha((DIRECTORY/'plan.json').read_bytes()),python=platform.python_version(),platform=platform.platform(),source_hashes={p:sha((ROOT/p).read_bytes()) for p in SOURCES})
    prior=checkpoint/'provenance.json'
    if prior.exists() and json.loads(prior.read_text())!=provenance:raise RuntimeError('Changed checkpoint provenance')
    prior.write_bytes(canonical(provenance))
    with zipfile.ZipFile(results/'execution-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in SOURCES:z.write(ROOT/p,p)
    binary=Path('/tmp/shake-performance-frozen-runner');shutil.copyfile(BINARY,binary);binary.chmod(0o700)
    scene=json.loads((ROOT/plan['scene']).read_text())['scene']
    references={name:json.loads((ROOT/'research/fast-shake-diagnostic'/file).read_text()) for name,file in [('fixed','fixed_1_25us_corrected.json'),('guard','eighth_travel.json')]}
    runs={};checks={}
    for name in plan['order']:
        lane=name.rsplit('_',1)[0];path=checkpoint/(name+'.json')
        if path.exists():result=json.loads(path.read_text())
        else:
            start=time.perf_counter()
            try:result=run(scene,binary=binary,**plan['common'],**plan['settings'][lane])
            except subprocess.CalledProcessError as e:result=dict(rejected=e.stderr.strip(),exit_code=e.returncode,elapsed_s=time.perf_counter()-start)
            path.write_bytes(canonical(result))
        runs[name]=result
        if 'rejected' in result:checks[name]=dict(qualified=False,rejected=result['rejected'])
        else:
            d=physical(scene,result);e={key:errors(result,r) for key,r in references.items()}
            good=all(d[k]<=v for k,v in plan['physical_gates'].items()) and all(error[k]<=v for error in e.values() for k,v in plan['trajectory_budget'].items())
            checks[name]=dict(qualified=good,physical=d,errors=e)
        print(name,checks[name]['qualified'],result.get('step_s',result.get('rejected')),flush=True)
    lanes={}
    for lane in plan['settings']:
        names=[name for name in plan['order'] if name.rsplit('_',1)[0]==lane];good=all(checks[n]['qualified'] for n in names)
        identical=good and all(runs[n]['states']==runs[names[0]]['states'] for n in names)
        native=[runs[n]['step_s'] for n in names if 'rejected' not in runs[n]];whole=[runs[n]['wall_time_s'] for n in names if 'rejected' not in runs[n]]
        lanes[lane]=dict(qualified=good and identical,bitwise_repeatable=identical,native_s=native,whole_process_s=whole,median_native_s=float(np.median(native)) if native else None,median_whole_process_s=float(np.median(whole)) if whole else None,updates=[runs[n]['collision_updates'] for n in names if 'rejected' not in runs[n]])
    passed=all(v['qualified'] for v in lanes.values())
    with zipfile.ZipFile(results/'traces.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name,result in runs.items():z.writestr(name+'.json',canonical(result))
    summary=dict(**provenance,checks=checks,lanes=lanes,qualified=passed,median_native_ratio=lanes['reference']['median_native_s']/lanes['candidate']['median_native_s'] if passed else None,hashes={p:sha((results/p).read_bytes()) for p in ['traces.zip','execution-source.zip']})
    (results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('DONE',summary['qualified'],summary['median_native_ratio'],flush=True)
if __name__=='__main__':main()
