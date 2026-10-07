"""Prospectively frozen shared-geometry reference ladder and cost comparison."""
import hashlib,json,platform,shutil,subprocess,time,zipfile
from pathlib import Path
import numpy as np
from spatial_engine import run,BINARY
from research.audit_fast_shake_diagnostic import errors,physical
ROOT=Path(__file__).parents[1];D=ROOT/'research/shared-shake-study'
SOURCES=['spatial_engine.py','spatial_backend/runner.cpp','spatial_backend/coulomb.h','spatial_backend/coulomb_polish.h','spatial_backend/newton_linear.h','spatial_backend/normal_qp.h','spatial_backend/shared_contact.h','spatial_backend/CMakeLists.txt','research/run_shared_shake.py','research/audit_shared_shake.py','research/shared-shake-study/plan.json','research/audit_fast_shake_diagnostic.py','research/fast-shake-diagnostic/scene.json']
def sha(x):return hashlib.sha256(x).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def summarize(plan,scene,runs):
    reference_names=list(plan['reference_levels']);reference_good=all('rejected' not in runs[n] for n in reference_names)
    edges={}
    for a,b in plan['reference_edges']:
        e=errors(runs[a],runs[b]) if 'rejected' not in runs[a] and 'rejected' not in runs[b] else None
        good=e is not None and all(e[k]<=v for k,v in plan['reference_budget'].items());edges[a+'->'+b]=dict(errors=e,qualified=good);reference_good&=good
    diagnostics={};checks={}
    for name,r in runs.items():
        if 'rejected' in r:checks[name]=False;continue
        d=physical(scene,r);diagnostics[name]=d
        good=all(d[k]<=v for k,v in plan['physical_gates'].items()) and r['coulomb_residual_max_m_s']<=plan['common']['contact_tolerance_m_s'] and r['coupled_fallbacks']==0
        if name in reference_names:reference_good&=good;checks[name]=good
        else:
            e=errors(r,runs[reference_names[-1]]) if 'rejected' not in runs[reference_names[-1]] else None
            checks[name]=dict(physical_pass=good,errors=e,qualified=good and e is not None and all(e[k]<=v for k,v in plan['trajectory_budget'].items()))
    lanes={}
    for lane in plan['settings']:
        names=[n for n in plan['order'] if n.rsplit('_',1)[0]==lane];good=all(isinstance(checks[n],dict) and checks[n]['qualified'] for n in names)
        identical=good and all(runs[n]['states']==runs[names[0]]['states'] for n in names)
        native=[runs[n]['step_s'] for n in names if 'rejected' not in runs[n]];whole=[runs[n]['wall_time_s'] for n in names if 'rejected' not in runs[n]]
        lanes[lane]=dict(qualified=good and identical,bitwise_repeatable=identical,native_s=native,whole_s=whole,median_native_s=float(np.median(native)) if native else None)
    qualified=bool(reference_good and all(v['qualified'] for v in lanes.values()))
    return dict(reference_qualified=bool(reference_good),reference_edges=edges,physical=diagnostics,checks=checks,lanes=lanes,qualified=qualified,median_native_ratio=lanes['reference']['median_native_s']/lanes['candidate']['median_native_s'] if qualified else None)
def main():
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();plan=json.loads((D/'plan.json').read_text());scene=json.loads((ROOT/plan['scene']).read_text())['scene']
    for p in SOURCES:assert subprocess.check_output(['git','show',source+':'+p],cwd=ROOT)==(ROOT/p).read_bytes(),p
    out=D/'results';out.mkdir(exist_ok=True);cp=out/'checkpoints';cp.mkdir(exist_ok=True)
    provenance=dict(execution_source_commit=source,source_hashes={p:sha((ROOT/p).read_bytes()) for p in SOURCES},binary_sha256=sha(BINARY.read_bytes()),plan_sha256=sha((D/'plan.json').read_bytes()),platform=platform.platform(),python=platform.python_version())
    prior=cp/'provenance.json'
    if prior.exists():assert json.loads(prior.read_text())==provenance
    prior.write_bytes(canonical(provenance))
    with zipfile.ZipFile(out/'execution-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in SOURCES:z.write(ROOT/p,p)
    binary=Path('/tmp/shared-shake-frozen-runner');shutil.copyfile(BINARY,binary);binary.chmod(0o700)
    runs={}
    for name in list(plan['reference_levels'])+plan['order']:
        controls=plan['reference_levels'][name] if name in plan['reference_levels'] else plan['settings'][name.rsplit('_',1)[0]];path=cp/(name+'.json')
        if path.exists():r=json.loads(path.read_text())
        else:
            before=time.perf_counter()
            try:r=run(scene,binary=binary,**plan['common'],**controls)
            except subprocess.CalledProcessError as e:r=dict(rejected=e.stderr.strip(),exit_code=e.returncode,elapsed_s=time.perf_counter()-before)
            path.write_bytes(canonical(r))
        runs[name]=r;print(name,r.get('step_s',r.get('rejected')),flush=True)
    with zipfile.ZipFile(out/'traces.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n,r in runs.items():z.writestr(n+'.json',canonical(r))
    summary=dict(**provenance,**summarize(plan,scene,runs),hashes={p:sha((out/p).read_bytes()) for p in ['traces.zip','execution-source.zip']})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('DONE',summary['qualified'],summary['median_native_ratio'],flush=True)
if __name__=='__main__':main()
