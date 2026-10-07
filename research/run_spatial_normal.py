"""Exactly solvable 3D native contact graphs and a fair matrix-elimination ablation."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import zipfile
import numpy as np
from spatial_engine import run,energy
from research.spatial_scenes import driven_row,touching_container

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'research/spatial-normal'
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenes(plan):
    result={}
    for config in plan['scenes']:
        args={k:v for k,v in config.items() if k not in ('id','kind')}
        if config['kind']=='packed':scene,half=touching_container(**args,epsilon_m=plan['geometry_tolerance_m'])
        else:
            scene=driven_row(**args);half=None;axis=config['axis'];epsilon=plan['geometry_tolerance_m']
            scene['bodies'][0]['position'][axis]+=epsilon
            for i,b in enumerate(scene['bodies'][1:]):b['position'][axis]=i*(.2-epsilon)
        result[config['id']]=dict(scene=scene,half=half)
    return result


def metrics(scene,r):
    state=np.asarray(r['states']);times=np.asarray(r['times']);U=np.asarray(scene['bodies'][0]['velocity']);count=len(scene['bodies'])-1
    exact=state[0,1:,:3]+times[:,None,None]*U
    work=count*np.dot(U,U);E=energy(r)
    return dict(max_position_error_m=float(np.max(np.linalg.norm(state[:,1:,:3]-exact,axis=2))),max_velocity_error_m_s=float(np.max(np.linalg.norm(state[1:,1:,7:10]-U,axis=2))),max_omega_rad_s=float(np.max(np.linalg.norm(state[:,:,10:13],axis=2))),max_contact_penetration_m=r['max_contact_penetration_m'],max_closing_contact_speed_m_s=r['max_closing_contact_speed_m_s'],max_container_surface_excess_m=r.get('max_container_surface_excess_m',0),boundary_work_error_J=float(abs(r['boundary_work_J']-work)),energy_balance_error_J=float(abs(E[-1]-.5*work)),coupled_fallbacks=r['coupled_fallbacks'],normal_qp_rejections=r['normal_qp_rejections'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=DIRECTORY);dest=parser.parse_args().directory
    plan=json.loads((DIRECTORY/'plan.json').read_text());authored=scenes(plan);source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Freeze source and plan first')
    checkpoint=dest/'results/checkpoints';checkpoint.mkdir(parents=True,exist_ok=True)
    provenance=dict(execution_source_commit=source,plan_sha256=sha(DIRECTORY/'plan.json'),binary_sha256=sha(ROOT/'build/spatial/spatial_runner'))
    path=checkpoint/'provenance.json'
    if path.exists() and json.loads(path.read_text())!=provenance:raise RuntimeError('Checkpoint provenance differs')
    path.write_bytes(canonical(provenance));traces={};summary={};warmups={}
    for name,item in authored.items():
        modes={}
        for mode,settings in plan['modes'].items():
            warm=run(item['scene'],dt=plan['dt_s'],**settings);warmups[f'{name}/{mode}']=dict(step_s=warm['step_s'],states_sha256=hashlib.sha256(canonical(warm['states'])).hexdigest());runs=[]
            for i in range(plan['retained_repetitions']):
                key=f'{name}/{mode}/{i}.json';path=checkpoint/key;path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():r=json.loads(path.read_text())
                else:r=run(item['scene'],dt=plan['dt_s'],**settings);path.write_bytes(canonical(r))
                traces[key]=r;runs.append(r);print(key,r['step_s'],metrics(item['scene'],r),flush=True)
            diagnostics=[metrics(item['scene'],r) for r in runs];passed=all(d[k]<=v for d in diagnostics for k,v in plan['analytic_gates'].items())
            modes[mode]=dict(qualified=passed,diagnostics=diagnostics,median_step_s=float(np.median([r['step_s'] for r in runs])),mobility_rows_max=runs[0]['mobility_rows_max'],mobility_matrix_bytes_max=runs[0]['mobility_matrix_bytes_max'],normal_qp_solves=runs[0]['normal_qp_solves'])
        identity=all(traces[f'{name}/compact/{i}.json']['states']==traces[f'{name}/postassembly/{i}.json']['states'] for i in range(3))
        qualified=identity and all(m['qualified'] for m in modes.values())
        summary[name]=dict(qualified=qualified,bitwise_identical_states=identity,modes=modes,speedup=modes['postassembly']['median_step_s']/modes['compact']['median_step_s'] if qualified else None,mobility_payload_reduction=modes['postassembly']['mobility_matrix_bytes_max']/modes['compact']['mobility_matrix_bytes_max'] if qualified else None)
    result=dest/'results'
    (result/'scenes.json').write_bytes(canonical(authored))
    with zipfile.ZipFile(result/'traces.zip','w',zipfile.ZIP_DEFLATED) as z:
        for key,r in traces.items():z.writestr(key,canonical(r))
    paths=['spatial_engine.py','spatial_backend/runner.cpp','spatial_backend/normal_qp.h','spatial_backend/CMakeLists.txt','research/spatial_scenes.py','research/run_spatial_normal.py','research/spatial-normal/plan.json']
    with zipfile.ZipFile(result/'execution-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(ROOT/p,p)
    data=dict(**provenance,platform=platform.platform(),python=platform.python_version(),warmups=warmups,history_count=len(traces),scenes=summary,hashes={p:sha(result/p) for p in ['scenes.json','traces.zip','execution-source.zip']},source_hashes={p:sha(ROOT/p) for p in paths})
    (result/'summary.json').write_text(json.dumps(data,indent=2)+'\n');print('DONE',source,len(traces),flush=True)

if __name__=='__main__':main()
