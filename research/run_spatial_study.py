"""Execute the immutable 3D protocol; checkpoint every retained history."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import zipfile
import numpy as np
from spatial_engine import run, errors, BULLET_COMMIT
from research.spatial_scenes import container, driven_row
from research.spatial_metrics import diagnostics

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'research/spatial-validation'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalized(metrics,budget):return max(metrics[k]/budget[k] for k in budget)


def scenes(plan):
    result={}
    for config in plan['scenes']:
        args={k:v for k,v in config.items() if k not in ('id','kind')}
        if config['kind']=='row':scene=driven_row(**args);half=None
        else:scene,half=container(**args)
        result[config['id']]=dict(scene=scene,half=half)
    return result


def physical_pass(diag,checks):
    keys=['quaternion_norm_error','energy_change_minus_boundary_work_J']
    if 'container_surface_excess_m' in diag:keys+=['container_surface_excess_m']
    if 'row_final_velocity_max_error_m_s' in diag:keys+=['row_final_velocity_max_error_m_s']
    return all(diag[k]<=checks[k] for k in keys)


def describe(scene,result,half):
    d=diagnostics(scene,result,half)
    if half is None:
        a=np.asarray(result['states']);d['row_final_velocity_max_error_m_s']=float(np.max(np.linalg.norm(a[-1,1:,7:10]-a[-1,0,7:10],axis=1)))
    return d


def analyze(plan,authored,traces):
    summary={};budget=plan['candidate_budget'];refbudget={k:v*plan['reference_budget_fraction'] for k,v in budget.items()}
    for name,item in authored.items():
        get=lambda mode,rep=0:traces[f'{name}/{mode}/{rep}.json']
        reference=get('reference');edges=[]
        for a,b in plan['reference_edges']:
            metrics=errors(get(b),get(a));edges.append(dict(modes=[a,b],errors=metrics,normalized=normalized(metrics,refbudget)))
        refdiag=describe(item['scene'],reference,item['half'])
        qualified=all(e['normalized']<=1 for e in edges) and physical_pass(refdiag,plan['physical_checks'])
        candidates={}
        for mode in plan['candidate_modes']:
            histories=[get(mode,i) for i in range(plan['timing']['candidate_retained_repetitions'])]
            metrics=errors(reference,histories[0]);diags=[describe(item['scene'],r,item['half']) for r in histories]
            passed=qualified and normalized(metrics,budget)<=1 and all(physical_pass(d,plan['physical_checks']) for d in diags)
            candidates[mode]=dict(errors=metrics,normalized=normalized(metrics,budget),accuracy_qualified=passed,median_step_s=float(np.median([r['step_s'] for r in histories])),diagnostics=diags[0],coupled_updates=histories[0]['coupled_updates'],sequential_updates=histories[0]['sequential_updates'],collision_updates=histories[0]['collision_updates'])
        choices=[m for m,c in candidates.items() if c['accuracy_qualified']]
        choice=min(choices,key=lambda m:candidates[m]['median_step_s']) if choices else None
        summary[name]=dict(reference_qualified=qualified,reference_edges=edges,reference_diagnostics=refdiag,candidates=candidates,choice=choice)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=DIRECTORY);args=parser.parse_args();dest=args.directory
    plan=json.loads((DIRECTORY/'plan.json').read_text());authored=scenes(plan);checkpoint=dest/'results/checkpoints';checkpoint.mkdir(parents=True,exist_ok=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    if dirty.strip():raise RuntimeError('Freeze and publish source/plan before executing')
    provenance=dict(execution_source_commit=source,plan_sha256=sha(DIRECTORY/'plan.json'),binary_sha256=sha(ROOT/'build/spatial/spatial_runner'))
    receipt=checkpoint/'provenance.json'
    if receipt.exists() and json.loads(receipt.read_text())!=provenance:raise RuntimeError('Checkpoint source/plan/binary differs; use a new directory')
    receipt.write_bytes(canonical(provenance))
    traces={};warmups={}
    for name,item in authored.items():
        for mode,settings in {**plan['reference_modes'],**plan['candidate_modes']}.items():
            candidate=mode in plan['candidate_modes'];count=plan['timing']['candidate_retained_repetitions'] if candidate else 1
            if candidate:
                warm=run(item['scene'],dt=plan['dt_s'],**settings);warmups[f'{name}/{mode}']=dict(step_s=warm['step_s'],states_sha256=hashlib.sha256(canonical(warm['states'])).hexdigest())
            for i in range(count):
                key=f'{name}/{mode}/{i}.json';path=checkpoint/key;path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():r=json.loads(path.read_text())
                else:
                    r=run(item['scene'],dt=plan['dt_s'],**settings)
                    path.write_bytes(canonical(r))
                traces[key]=r
                print(f'{key}: {r["step_s"]:.4f}s, {r["collision_updates"]} updates, {r["coupled_fallbacks"]} MLCP fallbacks',flush=True)
        (dest/'results/partial.json').write_bytes(canonical(analyze(plan,{name:item},{k:v for k,v in traces.items() if k.startswith(name+'/')})))
    result=dest/'results';result.mkdir(exist_ok=True)
    (result/'scenes.json').write_bytes(canonical(authored))
    with zipfile.ZipFile(result/'traces.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name,r in traces.items():z.writestr(name,canonical(r))
    paths=['spatial_engine.py','spatial_backend/runner.cpp','spatial_backend/CMakeLists.txt','research/spatial_scenes.py','research/spatial_metrics.py','research/run_spatial_study.py','research/spatial-validation/plan.json']
    with zipfile.ZipFile(result/'execution-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(ROOT/p,p)
    summary=dict(binary_sha256=provenance['binary_sha256'],compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0],execution_source_commit=source,plan_sha256=sha(DIRECTORY/'plan.json'),bullet_commit=BULLET_COMMIT,platform=platform.platform(),python=platform.python_version(),warmups=warmups,history_count=len(traces),scenes=analyze(plan,authored,traces),hashes={p:sha(result/p) for p in ['scenes.json','traces.zip','execution-source.zip']})
    (result/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('DONE',source,len(traces),flush=True)

if __name__=='__main__':main()
