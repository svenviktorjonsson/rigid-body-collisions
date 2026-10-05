"""Six prospective lanes with retained policies and one bounded search tail; mandatory final freeze."""
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

def source_paths(plan):
    fixed=['spatial_engine.py','spatial_fidelity.py','research/spatial_scenes.py','research/spatial_metrics.py',
           'spatial_backend/runner.cpp','spatial_backend/CMakeLists.txt',
           'research/hull-search-completion/runner.py','research/hull-search-completion/plan.json',
           'research/hull-search-completion/README.md',
           'research/audit_shared_hulls.py','research/audit_hull_active_completion.py']
    native=[str(p.relative_to(ROOT)) for extension in ('*.h','*.cpp') for p in (ROOT/'spatial_backend').glob(extension)]
    return sorted(set(fixed+native+plan['freeze_required_paths']))

def recovery_config(plan,require_ready=False):
    cfg=plan['numerical_recovery']
    if cfg['stage']!='after_all_existing_pipeline_failure' or cfg['enabled'] is not True:raise RuntimeError('Only the declared post-failure numerical tail is permitted')
    pending=plan['preparation_status']!='FINALIZED_FOR_ROOT_PUBLISHED_EXECUTION'
    if cfg['status'] not in ('ROOT_APPROVED_PENDING_INTEGRATION_NATIVE_CONFIG','ROOT_APPROVED_FROZEN_NATIVE_CONFIG'):raise RuntimeError('Unapproved selected search configuration')
    if cfg['wire_flag'] is not None or cfg['default_enabled'] is not True:raise RuntimeError('No additional API wire/default change is permitted; existing contact_recovery gates the tail')
    if cfg['installed_helper']!='spatial_backend/projection_more.h' or cfg['numerical_model_key']!='projection_tail_policy':raise RuntimeError('Root-selected helper/metadata contract changed')
    if cfg['approved_caps']!={'max_rows':64,'max_svd_calls':2048,'max_iteration_steps':2048} or any(type(v) is not int for v in cfg['approved_caps'].values()):raise RuntimeError('Exact bounded native caps changed')
    if not all(isinstance(cfg[k],str) and cfg[k] for k in ('strategy','method','final_gate')):raise RuntimeError('Exact selected strategy/method/final gate required')
    if cfg['installed_helper'] not in plan['freeze_required_paths']:raise RuntimeError('Selected helper absent from frozen sources')
    checks=cfg['preexecution_checks']
    if set(checks)!={'published_integration_source','native_build_ready','23_default_controls_and_22_endpoint_preservation'} or any(type(v) is not bool for v in checks.values()):raise RuntimeError('Root preexecution checks malformed')
    if not pending:
        if cfg['status']!='ROOT_APPROVED_FROZEN_NATIVE_CONFIG' or not all(checks.values()):raise RuntimeError('Root integration/build/default proof approval missing')
    elif plan['preparation_status']!='PENDING_ROOT_INTEGRATION_BUILD_AND_23_DEFAULT_PROOF':raise RuntimeError('Unknown pending status')
    if require_ready and pending:raise RuntimeError('Wait for root-published integration, BUILD READY and23-default/22-preservation proof')
    return cfg,pending

def frozen_artifact_hashes(plan,require_ready=False):
    directory=plan['preceding_protocol_results']
    required={directory+'/'+n for n in ['summary.json','scenes.json','traces.zip','execution-source.zip']}|{plan['preceding_protocol_plan']}
    preceding=plan['preceding_protocol_artifact_hashes']
    if plan['preceding_protocol_freeze_status']!='FINALIZED_SIX_ATTEMPT_ARCHIVE' or set(preceding)!=required:raise RuntimeError('Exactly five sealed predecessor artifacts required')
    terminal=json.loads((ROOT/(directory+'/summary.json')).read_text())
    if terminal['execution_source_commit']!=plan['preceding_protocol_source_commit'] or terminal['complete'] is not True or terminal['attempt_count']!=6 or terminal['planned_attempt_count']!=6:raise RuntimeError('Wrong or incomplete predecessor archive')
    hashes=dict(plan['baseline_artifact_hashes'],**plan['historical_gap_artifact_hashes'],**preceding)
    if any(not isinstance(h,str) or not re.fullmatch(r'[0-9a-f]{64}',h) for h in hashes.values()):raise RuntimeError('Malformed frozen archive hash')
    return hashes

def validate_plan(plan,require_ready=False):
    baseline=json.loads((ROOT/plan['baseline_plan']).read_text());previous=json.loads((ROOT/plan['preceding_protocol_plan']).read_text())
    cfg,pending=recovery_config(plan,require_ready)
    expected=dict(previous['common'])
    if plan['common']!=expected:raise RuntimeError('Original bca common settings or new tail wire changed')
    if plan['retained_numerical_changes']!=previous['declared_numerical_change'] or plan['declared_numerical_change']!={'bounded_projection_merit_tail':{'baseline':False,'candidate':True}}:raise RuntimeError('Undeclared numerical policy change')
    for key in ['dt_s','trajectory_budget','physical_gates','reference_rule','scenes','contact_point_policy']:
        if plan[key]!=baseline[key]:raise RuntimeError('Original52 scenes/gates changed: '+key)
    if plan['common']['position_stabilization']!='split_translation_combined' or plan['common']['early_component_recovery'] is not True or plan['declared_shape_cache_margin_order']['candidate']!='margin before recalc':raise RuntimeError('Prior bca policies not retained')
    if plan['candidates'] or plan['candidate_repetitions']!=0:raise RuntimeError('No candidate-cost trials allowed')
    if len(plan['scenes'])!=2 or any(c['fractions']!=[.06,.03,.015] or c['duration']!=.12 for c in plan['scenes']):raise RuntimeError('Exactly six original full-duration lanes required')
    if plan['baseline_source_commit']!='52f7e6d244e92a8405134e0222ce14fc3eda0ef6' or plan['historical_gap_source_commit']!='108a9bb4c7899f75d760b27b179cc56557904a08' or plan['preceding_protocol_source_commit']!='bca35c3103a78c731b37ea8a467e5fc71d13e8aa':raise RuntimeError('Wrong predecessor source identities')
    if plan['preceding_archive_publication_commit']!='3bfc51108b89ba2f7965d33ec3104a3ac4ce8451':raise RuntimeError('Wrong sealed predecessor publication identity')
    for path,h in frozen_artifact_hashes(plan,require_ready).items():
        if sha((ROOT/path).read_bytes())!=h:raise RuntimeError('Frozen archive changed: '+path)
    if require_ready:
        for path in source_paths(plan):
            if not (ROOT/path).is_file():raise RuntimeError('Missing frozen native/source path: '+path)
    oldscenes=json.loads((ROOT/plan['baseline_scenes']).read_text());authored={}
    for config in plan['scenes']:
        scene,half=container(**{k:v for k,v in config.items() if k not in ('id','fractions')})
        entry=dict(scene=scene,half=half)
        if entry!=oldscenes[config['id']]:raise RuntimeError('Original52 authored scene changed')
        authored[config['id']]=entry
    return authored

def runtime_libraries():
    linked=subprocess.check_output(['ldd',str(ROOT/'build/spatial/spatial_runner')],text=True)
    return {str(Path(path).resolve()):sha(Path(path).resolve().read_bytes())
            for path in re.findall(r'(/\S+)\s+\(',linked)}

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
        contact=np.isfinite(result['coulomb_residual_max_m_s']) and 0<=result['coulomb_residual_max_m_s']<=plan['common']['contact_tolerance_m_s']
        position_solves=result.get('translation_split_solves')
        position_residual=result.get('translation_split_residual_max_m_s')
        position=bool(isinstance(position_solves,int) and not isinstance(position_solves,bool) and position_solves>=0
                      and position_residual is not None and np.isfinite(position_residual)
                      and 0<=position_residual<=plan['common']['contact_tolerance_m_s']
                      and (position_solves>0 or position_residual==0))
        shared=result.get('contact_point_policy')=='shared' and result.get('numerical_model',{}).get('contact_point_policy')=='shared'
        eligible[lane]=bool(finite and gates and contact and position and shared)
    edges=[]
    for left,right in [('reference_0','reference_1'),('reference_1','reference_2')]:
        error=errors(runs[left],runs[right]) if eligible.get(left) and eligible.get(right) else None
        passed=bool(error and all(np.isfinite(error[k]) and error[k]<=limit/4 for k,limit in plan['trajectory_budget'].items()))
        edges.append(dict(left=left,right=right,passed=passed,error=error))
    return dict(reference_qualified=all(e['passed'] for e in edges),reference='reference_2',edges=edges,
                physical_eligible=eligible,diagnostics=physical,candidates={},choice=None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit');parser.add_argument('--check-plan',action='store_true');parser.add_argument('--build-ready',action='store_true')
    parser.add_argument('--workload-note',default='Parallel collaborative workload; descriptive timings only')
    args=parser.parse_args();plan=json.loads(PLAN.read_text());authored=validate_plan(plan)
    if args.check_plan:
        _,pending=recovery_config(plan)
        print('PENDING selected bounded projection tail; root integration/BUILD READY/23-default proof required; six original52 lanes and sealed bca/gap archives validated; no native execution' if pending else 'READY prospective bounded-search protocol; execution requires published integration SHA AND explicit BUILD READY; no native execution')
        return
    authored=validate_plan(plan,require_ready=True)
    if not args.build_ready:parser.error('--build-ready operator attestation required after root declares native BUILD READY')
    for key,value in plan['thread_environment'].items():
        if os.environ.get(key)!=value:raise RuntimeError('Set '+key+'='+value+' before native study execution')
    if not args.source_commit:parser.error('--source-commit is mandatory; wait for the supplied frozen integration SHA')
    if not re.fullmatch(r'[0-9a-f]{40}',args.source_commit):parser.error('--source-commit must be the exact full40-character root-published SHA')
    source=subprocess.check_output(['git','rev-parse',args.source_commit],cwd=ROOT,text=True).strip();paths=source_paths(plan)
    if source!=args.source_commit:raise RuntimeError('Source SHA did not resolve exactly')
    binary_hash=sha((ROOT/'build/spatial/spatial_runner').read_bytes());library_hashes=runtime_libraries();guard(source,paths,binary_hash,library_hashes)
    directory=DIRECTORY/'results';checkpoint=directory/'checkpoints'
    provenance=dict(build_ready_attested=True,numerical_recovery=plan['numerical_recovery'],execution_source_commit=source,plan_sha256=sha(PLAN.read_bytes()),binary_sha256=binary_hash,
                    source_hashes={path:sha((ROOT/path).read_bytes()) for path in paths},runtime_library_hashes=library_hashes,workload_note=args.workload_note,
                    baseline_artifact_hashes=frozen_artifact_hashes(plan,True),declared_numerical_change=plan['declared_numerical_change'],preceding_protocol_source_commit=plan['preceding_protocol_source_commit'],thread_environment={key:os.environ.get(key) for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
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
                     accuracy_scope='Only complete lanes passing original physical gates and BOTH quarter-trajectory edges qualify; solver gates alone confer no accuracy',
                     interpretation='Prior three numerical policies retained; added declared bounded post-failure numerical-search tail; no law/gate relaxation, candidate-cost trials, causal attribution or speed ranking',
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
            for frozen_path,expected_hash in frozen_artifact_hashes(plan,True).items():
                if sha((ROOT/frozen_path).read_bytes())!=expected_hash:raise RuntimeError('Frozen archive changed before attempt: '+frozen_path)
            if path.exists():result=json.loads(path.read_text())
            else:
                if dump.exists() or progress.exists() or (directory/'attempts'/key).exists():raise RuntimeError('Uncheckpointed native/started-attempt artifacts survive; preserve and attest interruption, never overwrite/retry: '+key)
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
            for baseline_path,expected_hash in frozen_artifact_hashes(plan,True).items():
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
