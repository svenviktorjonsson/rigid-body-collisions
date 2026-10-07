"""Frozen-source circular 3D friction evidence; checkpoint every attempt."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import zipfile
import numpy as np
from spatial_engine import run
from spatial_fidelity import qualify
from research.spatial_scenes import container
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'research/spatial-friction-recovery384'
SOURCES=['spatial_engine.py','spatial_fidelity.py','spatial_backend/runner.cpp',
 'spatial_backend/coulomb.h','spatial_backend/coulomb_polish.h','spatial_backend/newton_linear.h','spatial_backend/normal_qp.h','spatial_backend/CMakeLists.txt',
 'spatial_backend/friction_checks.cpp','research/spatial_scenes.py','research/spatial_metrics.py',
 'research/run_spatial_recovery384.py','research/spatial-friction-recovery384/plan.json']
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def digest(b):return hashlib.sha256(b).hexdigest()

def main():
    plan=json.loads((DIRECTORY/'plan.json').read_text());source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in SOURCES:
        committed=subprocess.check_output(['git','show',f'{source}:{path}'],cwd=ROOT)
        if committed!=(ROOT/path).read_bytes():raise RuntimeError(f'Freeze source before execution: {path}')
    result=DIRECTORY/'results';checkpoint=result/'checkpoints';checkpoint.mkdir(parents=True,exist_ok=True)
    provenance=dict(execution_source_commit=source,plan_sha256=digest((DIRECTORY/'plan.json').read_bytes()),binary_sha256=digest((ROOT/'build/spatial/spatial_runner').read_bytes()))
    p=checkpoint/'provenance.json'
    if p.exists() and json.loads(p.read_text())!=provenance:raise RuntimeError('Provenance changed')
    p.write_bytes(canonical(provenance));all_runs={};authored={};receipts={}
    for config in plan['scenes']:
        name=config['id'];scene,half=container(**{k:v for k,v in config.items() if k not in ('id','fractions')});authored[name]=dict(scene=scene,half=half)
        lanes={f'reference_{i}':fraction for i,fraction in enumerate(config['fractions'])};lanes.update({f'{candidate}_{i}':fraction for candidate,fraction in plan['candidates'].items() for i in range(plan['candidate_repetitions'])})
        runs={}
        for lane,fraction in lanes.items():
            path=checkpoint/f'{name}/{lane}.json';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():r=json.loads(path.read_text())
            else:
                start=time.perf_counter()
                try:r=run(scene,dt=plan['dt_s'],travel_fraction=fraction,**plan['common'])
                except subprocess.CalledProcessError as e:r=dict(rejected=e.stderr.strip(),exit_code=e.returncode,elapsed_s=time.perf_counter()-start)
                path.write_bytes(canonical(r))
            all_runs[f'{name}/{lane}.json']=r;runs[lane]=r
            print(name,lane,'REJECT '+r['rejected'] if 'rejected' in r else f"{r['step_s']:.4f}s residual={r['coulomb_residual_max_m_s']:.3g} sweeps={r['coulomb_sweeps_max']}",flush=True)
        reference_names=[f'reference_{i}' for i in range(len(config['fractions']))]
        candidate_names=[f'{candidate}_{i}' for candidate in plan['candidates'] for i in range(plan['candidate_repetitions'])]
        q=qualify(scene,runs,reference_names,candidate_names,budget=plan['trajectory_budget'])
        candidates={}
        for candidate in plan['candidates']:
            names=[f'{candidate}_{i}' for i in range(plan['candidate_repetitions'])]
            good=all(q['candidates'][key]['qualified'] for key in names)
            native=[runs[key]['step_s'] for key in names if 'rejected' not in runs[key]]
            identity=len(native)==len(names) and all(runs[key]['states']==runs[names[0]]['states'] for key in names)
            candidates[candidate]=dict(qualified=good and identity,deterministic_states=identity,
                median_step_s=float(np.median(native)) if native else None,
                errors=[q['candidates'][key]['error'] for key in names],
                native_fast_solve_fraction=[runs[key]['coulomb_fast_solves']/max(1,runs[key]['coulomb_solves']) for key in names if 'rejected' not in runs[key]])
        passing=[key for key,v in candidates.items() if v['qualified']]
        choice=min(passing,key=lambda key:candidates[key]['median_step_s']) if passing else None
        q.update(candidates=candidates,choice=choice);receipts[name]=q
        print('RESULT',name,'reference',q['reference_qualified'],'choice',choice,flush=True)
    (result/'scenes.json').write_bytes(canonical(authored))
    with zipfile.ZipFile(result/'traces.zip','w',zipfile.ZIP_DEFLATED) as z:
        for key,r in all_runs.items():z.writestr(key,canonical(r))
    with zipfile.ZipFile(result/'execution-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in SOURCES:z.write(ROOT/path,path)
    data=dict(**provenance,platform=platform.platform(),python=platform.python_version(),attempt_count=len(all_runs),history_count=sum('rejected' not in r for r in all_runs.values()),scenes=receipts,
        hashes={p:digest((result/p).read_bytes()) for p in ['traces.zip','scenes.json','execution-source.zip']},source_hashes={p:digest((ROOT/p).read_bytes()) for p in SOURCES})
    (result/'summary.json').write_text(json.dumps(data,indent=2)+'\n');print('DONE',source,data['attempt_count'],data['history_count'],flush=True)
if __name__=='__main__':main()
