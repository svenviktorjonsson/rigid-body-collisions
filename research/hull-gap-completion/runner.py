"""Six prospectively paired signed-gap repair lanes; mandatory source freeze."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
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
           'research/hull-gap-completion/runner.py','research/hull-gap-completion/plan.json','research/hull-gap-completion/README.md',
           'research/audit_hull_gap_completion.py','research/audit_shared_hulls.py','research/audit_hull_active_completion.py']
    return sorted(set(fixed+[str(p.relative_to(ROOT)) for p in (ROOT/'spatial_backend').glob('*.h')]))

def validate_plan(plan):
    baseline=json.loads((ROOT/plan['baseline_plan']).read_text())
    expected=dict(baseline['common']);expected['position_stabilization']='split_translation_gap'
    if plan['common']!=expected or plan['declared_numerical_change']!={'position_stabilization':{'baseline':'split_translation','candidate':'split_translation_gap'}}:raise RuntimeError('Only the declared signed-gap translation repair may differ')
    for key in ['dt_s','trajectory_budget','physical_gates','reference_rule','scenes']:
        if plan[key]!=baseline[key]:raise RuntimeError('Paired protocol changed: '+key)
    if plan['contact_point_policy']!='shared':raise RuntimeError('This protocol requires corrected shared geometry')
    if plan['candidates'] or plan['candidate_repetitions']!=0:raise RuntimeError('This study has no candidate-selection trials')
    if len(plan['scenes'])!=2 or any(c['fractions']!=[.06,.03,.015] for c in plan['scenes']):raise RuntimeError('Exactly six prescribed reference attempts required')
    for path,expected_hash in plan['baseline_artifact_hashes'].items():
        if sha((ROOT/path).read_bytes())!=expected_hash:raise RuntimeError('Frozen baseline archive changed: '+path)
    if plan['baseline_source_commit']!='52f7e6d244e92a8405134e0222ce14fc3eda0ef6':raise RuntimeError('Wrong paired baseline SHA')
    oldscenes=json.loads((ROOT/plan['baseline_scenes']).read_text());authored={}
    for config in plan['scenes']:
        scene,half=container(**{k:v for k,v in config.items() if k not in ('id','fractions')})
        entry=dict(scene=scene,half=half)
        if entry!=oldscenes[config['id']]:raise RuntimeError('Authored scene changed: '+config['id'])
        authored[config['id']]=entry
    return authored

def runtime_libraries():
    linked=subprocess.check_output(['ldd',str(ROOT/'build/spatial/spatial_runner')],text=True)
    return {str(Path(path).resolve()):sha(Path(path).resolve().read_bytes())
            for path in re.findall(r'^\s*\S+\s+=>\s+(/\S+)',linked,re.M)}

def guard(source,paths,binary_hash,library_hashes):
    for path in paths:
        committed=subprocess.check_output(['git','show',f'{source}:{path}'],cwd=ROOT)
        if committed!=(ROOT/path).read_bytes():raise RuntimeError('Source changed from frozen integration SHA: '+path)
    if sha((ROOT/'build/spatial/spatial_runner').read_bytes())!=binary_hash:raise RuntimeError('Native binary changed during study')
    for path,expected in library_hashes.items():
        if sha(Path(path).read_bytes())!=expected:raise RuntimeError('Native runtime library changed during study: '+path)

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
    if args.check_plan:print('Plan valid: six exact baseline52 scenes/settings; only signed-gap translation repair differs; no native execution');return
    for key,value in plan['thread_environment'].items():
        if os.environ.get(key)!=value:raise RuntimeError('Set '+key+'='+value+' before native study execution')
    if not args.source_commit:parser.error('--source-commit is mandatory; wait for the supplied frozen integration SHA')
    if not re.fullmatch(r'[0-9a-f]{40}',args.source_commit):parser.error('--source-commit must be the exact full40-character root-published SHA')
    source=subprocess.check_output(['git','rev-parse',args.source_commit],cwd=ROOT,text=True).strip();paths=source_paths()
    if source!=args.source_commit:raise RuntimeError('Source SHA did not resolve exactly')
    binary_hash=sha((ROOT/'build/spatial/spatial_runner').read_bytes());library_hashes=runtime_libraries();guard(source,paths,binary_hash,library_hashes)
    directory=DIRECTORY/'results';checkpoint=directory/'checkpoints'
    provenance=dict(execution_source_commit=source,plan_sha256=sha(PLAN.read_bytes()),binary_sha256=binary_hash,
                    source_hashes={path:sha((ROOT/path).read_bytes()) for path in paths},runtime_library_hashes=library_hashes,workload_note=args.workload_note,
                    baseline_artifact_hashes=plan['baseline_artifact_hashes'],thread_environment={key:os.environ.get(key) for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
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
                     attempt_count=len(all_runs),planned_attempt_count=6,complete=len(all_runs)==6 and not any(r.get('attempt_status')=='interrupted' for r in all_runs.values()),
                     interruption_count=sum(r.get('attempt_status')=='interrupted' for r in all_runs.values()),
                     history_count=sum('rejected' not in r for r in all_runs.values()),scenes=receipts,
                     rejection_diagnostics=rejections,changed_discretization=plan['changed_discretization'],
                     native_ledger_receipts={key:dict(path='ledgers/'+key,sha256=sha((directory/'ledgers'/key).read_bytes())) for key in all_runs if (directory/'ledgers'/key).exists()},
                     hashes={path:sha((directory/path).read_bytes()) for path in ['scenes.json','traces.zip','execution-source.zip']})
        atomic(directory/'summary.json',json.dumps(summary,indent=2,allow_nan=False).encode()+b'\n')
    for config in plan['scenes']:
        name=config['id'];scene=authored[name]['scene'];runs={}
        for i,fraction in enumerate(config['fractions']):
            lane=f'reference_{i}';key=f'{name}/{lane}.json';path=checkpoint/key
            dump=directory/'rejections'/name/(lane+'.json')
            progress=directory/'progress'/name/(lane+'.json')
            guard(source,paths,binary_hash,library_hashes)
            if path.exists():result=json.loads(path.read_text())
            else:
                if dump.exists() or progress.exists():raise RuntimeError('Uncheckpointed native artifacts survive; preserve and attest interruption, never overwrite/retry: '+key)
                atomic(directory/'attempts'/key,canonical(dict(status='started',scene=name,lane=lane,travel_fraction=fraction,execution_source_commit=source,runner_pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))))
                dump.parent.mkdir(parents=True,exist_ok=True);progress.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
                try:result=run(scene,dt=plan['dt_s'],travel_fraction=fraction,rejected_contact_path=str(dump),progress_checkpoint_path=str(progress),contact_point_policy=plan['contact_point_policy'],**plan['common'])
                except subprocess.CalledProcessError as e:
                    result=dict(rejected=(e.stderr or '').strip(),exit_code=e.returncode,elapsed_s=time.perf_counter()-start,attempt_status='interrupted' if e.returncode<0 else 'engine_rejected')
                    if e.returncode<0:
                        result.update(complete=False,interruption_signal=-e.returncode,interruption='Observed native process terminated by signal; not a physical infeasibility claim')
                except KeyboardInterrupt:
                    result=dict(rejected='Driver KeyboardInterrupt observed; native terminal status unavailable',attempt_status='interrupted',complete=False,driver_exception='KeyboardInterrupt',elapsed_s=time.perf_counter()-start)
                if 'rejected' in result:
                    result['rejection_dump']=str(dump.relative_to(directory)) if dump.exists() else None
                    result['rejection_dump_status']='captured' if dump.exists() else 'engine rejected without matrix snapshot; retained as-is'
                if progress.exists():result['native_progress']=dict(path=str(progress.relative_to(directory)),sha256=sha(progress.read_bytes()))
                if 'rejected' not in result:result['attempt_status']='history_complete'
                atomic(path,canonical(result))
            all_runs[key]=result;runs[lane]=result
            guard(source,paths,binary_hash,library_hashes)
            for baseline_path,expected_hash in plan['baseline_artifact_hashes'].items():
                if sha((ROOT/baseline_path).read_bytes())!=expected_hash:raise RuntimeError('Baseline archive changed during study: '+baseline_path)
            if dump.exists():
                data=json.loads(dump.read_text());rejections[key]=dict(path=str(dump.relative_to(directory)),sha256=sha(dump.read_bytes()),
                      phase=data['phase'],rows=len(data['b']),residual_m_s=data['residual_m_s'],tolerance_m_s=data['tolerance_m_s'])
            ledger={key:result[key] for key in plan['native_ledger_fields'] if key in result}
            prefix_ledger={}
            if progress.exists():
                snapshot=json.loads(progress.read_text());prefix_ledger={key:snapshot[key] for key in plan['native_ledger_fields'] if key in snapshot}
            atomic(directory/'ledgers'/key,canonical(dict(final=ledger,native_prefix=prefix_ledger,final_available=len(ledger)==len(plan['native_ledger_fields']),prefix_available=len(prefix_ledger)==len(plan['native_ledger_fields']),scope=plan['ledger_scope'])))
            archive_progress()
            print('LEDGER',name,lane,json.dumps(dict(final=ledger,native_prefix=prefix_ledger),sort_keys=True),flush=True)
            print(name,lane,'REJECT '+result['rejected'] if 'rejected' in result else f"{result['step_s']:.4f}s residual={result['coulomb_residual_max_m_s']:.3g}",flush=True)
            if result.get('attempt_status')=='interrupted':
                receipts[name]=qualify(plan,scene,runs);archive_progress()
                print('INTERRUPTED',source,key,'actual observation retained; remaining attempts unexecuted',flush=True);return
        receipts[name]=qualify(plan,scene,runs);archive_progress()
        print('REFERENCE',name,receipts[name]['reference_qualified'],flush=True)
    guard(source,paths,binary_hash,library_hashes);archive_progress()
    print('DONE',source,len(all_runs),sum('rejected' not in r for r in all_runs.values()),flush=True)
if __name__=='__main__':main()
