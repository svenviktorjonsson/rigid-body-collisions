"""Predeclared six-lane shared-contact experiment; mandatory source SHA freeze."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from spatial_engine import run,errors
from research.spatial_metrics import diagnostics
from research.spatial_scenes import container
DIRECTORY=Path(__file__).resolve().parent
PLAN=DIRECTORY/'plan.json'

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(value):return hashlib.sha256(value).hexdigest()
def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(data);temporary.replace(path)

def source_paths():
    fixed=['spatial_engine.py','spatial_fidelity.py','research/spatial_scenes.py','research/spatial_metrics.py',
           'spatial_backend/runner.cpp','spatial_backend/CMakeLists.txt','spatial_backend/friction_checks.cpp',
           'research/shared-hull-followup/runner.py','research/shared-hull-followup/plan.json']
    return sorted(set(fixed+[str(p.relative_to(ROOT)) for p in (ROOT/'spatial_backend').glob('*.h')]))

def validate_plan(plan):
    baseline=json.loads((ROOT/plan['baseline_plan']).read_text())
    for key in ['common','dt_s','trajectory_budget','physical_gates','reference_rule','scenes']:
        if plan[key]!=baseline[key]:raise RuntimeError('Paired protocol changed: '+key)
    if plan['contact_point_policy']!='shared':raise RuntimeError('This protocol requires corrected shared geometry')
    if plan['candidates'] or plan['candidate_repetitions']!=0:raise RuntimeError('This study has no candidate-selection trials')
    if len(plan['scenes'])!=2 or any(c['fractions']!=[.06,.03,.015] for c in plan['scenes']):raise RuntimeError('Exactly six prescribed reference attempts required')
    oldscenes=json.loads((ROOT/plan['baseline_scenes']).read_text());authored={}
    for config in plan['scenes']:
        scene,half=container(**{k:v for k,v in config.items() if k not in ('id','fractions')})
        entry=dict(scene=scene,half=half)
        if entry!=oldscenes[config['id']]:raise RuntimeError('Authored scene changed: '+config['id'])
        authored[config['id']]=entry
    return authored

def guard(source,paths,binary_hash):
    for path in paths:
        committed=subprocess.check_output(['git','show',f'{source}:{path}'],cwd=ROOT)
        if committed!=(ROOT/path).read_bytes():raise RuntimeError('Source changed from frozen integration SHA: '+path)
    if sha((ROOT/'build/spatial/spatial_runner').read_bytes())!=binary_hash:raise RuntimeError('Native binary changed during study')

def qualify(plan,scene,runs):
    """Recompute all gates and both refinement edges without receipt reuse."""
    eligible={};physical={}
    for lane,result in runs.items():
        if 'rejected' in result:eligible[lane]=False;continue
        d=diagnostics(scene,result,scene['container_interior_half_extents_m'][0]);physical[lane]=d
        finite=all(np.isfinite(value) for value in d.values())
        gates=all(key in d and d[key]<=limit for key,limit in plan['physical_gates'].items())
        contact=np.isfinite(result['coulomb_residual_max_m_s']) and result['coulomb_residual_max_m_s']<=plan['common']['contact_tolerance_m_s']
        shared=result.get('contact_point_policy')=='shared' and result.get('numerical_model',{}).get('contact_point_policy')=='shared'
        eligible[lane]=bool(finite and gates and contact and shared)
    edges=[]
    for left,right in [('reference_0','reference_1'),('reference_1','reference_2')]:
        error=errors(runs[left],runs[right]) if eligible.get(left) and eligible.get(right) else None
        passed=bool(error and all(np.isfinite(error[k]) and error[k]<=limit/4 for k,limit in plan['trajectory_budget'].items()))
        edges.append(dict(left=left,right=right,passed=passed,error=error))
    return dict(reference_qualified=all(e['passed'] for e in edges),reference='reference_2',edges=edges,
                physical_eligible=eligible,diagnostics=physical,candidates={},choice=None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit');parser.add_argument('--check-plan',action='store_true')
    parser.add_argument('--workload-note',default='Parallel collaborative workload; descriptive timings only')
    args=parser.parse_args();plan=json.loads(PLAN.read_text());authored=validate_plan(plan)
    if args.check_plan:print('Plan valid: six exact paired scenes/settings; no native execution');return
    if not args.source_commit:parser.error('--source-commit is mandatory; wait for the supplied frozen integration SHA')
    source=subprocess.check_output(['git','rev-parse',args.source_commit],cwd=ROOT,text=True).strip();paths=source_paths()
    binary_hash=sha((ROOT/'build/spatial/spatial_runner').read_bytes());guard(source,paths,binary_hash)
    directory=DIRECTORY/'results';checkpoint=directory/'checkpoints'
    provenance=dict(execution_source_commit=source,plan_sha256=sha(PLAN.read_bytes()),binary_sha256=binary_hash,
                    source_hashes={path:sha((ROOT/path).read_bytes()) for path in paths},workload_note=args.workload_note)
    p=checkpoint/'provenance.json'
    if p.exists() and json.loads(p.read_text())!=provenance:raise RuntimeError('Checkpoint provenance changed; never overwrite an earlier study')
    atomic(p,canonical(provenance));atomic(directory/'scenes.json',canonical(authored))
    if not (directory/'execution-source.zip').exists():
        with zipfile.ZipFile(directory/'execution-source.zip.tmp','w',zipfile.ZIP_DEFLATED) as archive:
            for path in paths:archive.write(ROOT/path,path)
        (directory/'execution-source.zip.tmp').replace(directory/'execution-source.zip')
    with zipfile.ZipFile(directory/'execution-source.zip') as archive:
        if set(archive.namelist())!=set(paths) or any(sha(archive.read(path))!=provenance['source_hashes'][path] for path in paths):raise RuntimeError('Archived source does not match frozen source')
    all_runs={};receipts={};rejections={}
    def archive_progress():
        with zipfile.ZipFile(directory/'traces.zip.tmp','w',zipfile.ZIP_DEFLATED) as archive:
            for key,result in all_runs.items():archive.writestr(key,canonical(result))
        (directory/'traces.zip.tmp').replace(directory/'traces.zip')
        summary=dict(**provenance,platform=platform.platform(),python=platform.python_version(),
                     attempt_count=len(all_runs),planned_attempt_count=6,complete=len(all_runs)==6,
                     history_count=sum('rejected' not in r for r in all_runs.values()),scenes=receipts,
                     rejection_diagnostics=rejections,changed_discretization=plan['changed_discretization'],
                     hashes={path:sha((directory/path).read_bytes()) for path in ['scenes.json','traces.zip','execution-source.zip']})
        atomic(directory/'summary.json',json.dumps(summary,indent=2,allow_nan=False).encode()+b'\n')
    for config in plan['scenes']:
        name=config['id'];scene=authored[name]['scene'];runs={}
        for i,fraction in enumerate(config['fractions']):
            lane=f'reference_{i}';key=f'{name}/{lane}.json';path=checkpoint/key
            dump=directory/'rejections'/name/(lane+'.json')
            guard(source,paths,binary_hash)
            if path.exists():result=json.loads(path.read_text())
            else:
                dump.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
                try:result=run(scene,dt=plan['dt_s'],travel_fraction=fraction,rejected_contact_path=str(dump),contact_point_policy=plan['contact_point_policy'],**plan['common'])
                except subprocess.CalledProcessError as e:
                    result=dict(rejected=e.stderr.strip(),exit_code=e.returncode,elapsed_s=time.perf_counter()-start)
                if 'rejected' in result:
                    result['rejection_dump']=str(dump.relative_to(directory)) if dump.exists() else None
                    result['rejection_dump_status']='captured' if dump.exists() else 'engine rejected without matrix snapshot; retained as-is'
                atomic(path,canonical(result))
            all_runs[key]=result;runs[lane]=result
            if dump.exists():
                data=json.loads(dump.read_text());rejections[key]=dict(path=str(dump.relative_to(directory)),sha256=sha(dump.read_bytes()),
                      phase=data['phase'],rows=len(data['b']),residual_m_s=data['residual_m_s'],tolerance_m_s=data['tolerance_m_s'])
            archive_progress()
            print(name,lane,'REJECT '+result['rejected'] if 'rejected' in result else f"{result['step_s']:.4f}s residual={result['coulomb_residual_max_m_s']:.3g}",flush=True)
        receipts[name]=qualify(plan,scene,runs);archive_progress()
        print('REFERENCE',name,receipts[name]['reference_qualified'],flush=True)
    guard(source,paths,binary_hash);archive_progress()
    print('DONE',source,len(all_runs),sum('rejected' not in r for r in all_runs.values()),flush=True)
if __name__=='__main__':main()
