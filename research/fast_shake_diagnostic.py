"""Exploratory unchanged-law fast-shake controls; every attempt is retained."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'research/fast-shake-diagnostic'
SOURCES=['spatial_engine.py','spatial_backend/runner.cpp','spatial_backend/coulomb.h',
         'spatial_backend/normal_qp.h','research/fast_shake_diagnostic.py',
         'research/fast-shake-diagnostic/plan.json']
BINARY=Path('/tmp/fast-shake-diagnostic-runner')

def digest(value): return hashlib.sha256(value).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def prepare():
    plan=json.loads((DIRECTORY/'plan.json').read_text())
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    hashes={path:digest((ROOT/path).read_bytes()) for path in SOURCES}
    with zipfile.ZipFile(DIRECTORY/'execution-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in SOURCES: archive.write(ROOT/path,path)
    for path in SOURCES[:4]:
        if subprocess.check_output(['git','show',source+':'+path],cwd=ROOT)!=(ROOT/path).read_bytes():
            raise RuntimeError('Backend source is not frozen: '+path)
    metadata=dict(backend_source_commit=source,diagnostic_scripts='Exact snapshots included; may precede their publishing commit',
                  source_hashes=hashes,binary_sha256=digest(BINARY.read_bytes()),
                  scope=plan['scope'])
    save(DIRECTORY/'provenance.json',metadata)
    # Freeze the adapter as well as the executable; concurrent root changes
    # cannot alter these control trials after preparation.
    shutil.copyfile(ROOT/'spatial_engine.py','/tmp/fast-shake-diagnostic-adapter.py')

def adapter():
    spec=importlib.util.spec_from_file_location('fast_shake_frozen_adapter','/tmp/fast-shake-diagnostic-adapter.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def run_lane(lane):
    plan=json.loads((DIRECTORY/'plan.json').read_text()); provenance=json.loads((DIRECTORY/'provenance.json').read_text())
    if digest(BINARY.read_bytes())!=provenance['binary_sha256']:raise RuntimeError('Frozen binary changed')
    if digest(Path('/tmp/fast-shake-diagnostic-adapter.py').read_bytes())!=provenance['source_hashes']['spatial_engine.py']:
        raise RuntimeError('Frozen adapter changed')
    authored=json.loads((ROOT/'research/spatial-friction/results/scenes.json').read_text())['fast_shake27_spheres']['scene']
    start=time.perf_counter()
    try:
        result=adapter().run(authored,binary=BINARY,**plan['common'],**plan['runs'][lane])
    except subprocess.CalledProcessError as error:
        result=dict(rejected=error.stderr.strip(),exit_code=error.returncode)
    result['diagnostic_elapsed_s']=time.perf_counter()-start
    save(DIRECTORY/(lane+'.json'),result)
    print(lane,'rejected '+result['rejected'] if 'rejected' in result else
          f"updates={result['collision_updates']} native_s={result['step_s']:.3f}",flush=True)

def summarize():
    engine=adapter(); plan=json.loads((DIRECTORY/'plan.json').read_text())
    with zipfile.ZipFile(ROOT/'research/spatial-friction/results/traces.zip') as archive:
        baseline=json.loads(archive.read('fast_shake27_spheres/reference_4.json'))
        preceding=json.loads(archive.read('fast_shake27_spheres/reference_3.json'))
    runs={};report={}
    for lane in plan['runs']:
        path=DIRECTORY/(lane+'.json')
        if not path.exists():continue
        result=json.loads(path.read_text());runs[lane]=result
        if 'rejected' in result:report[lane]=dict(rejected=result['rejected']);continue
        matching=[]
        for t in baseline['times']:
            indices=[i for i,value in enumerate(result['times']) if abs(value-t)<1e-10]
            if len(indices)!=1:raise RuntimeError('No unique common sample')
            matching.append(indices[0])
        aligned=dict(result,times=baseline['times'],states=[result['states'][i] for i in matching])
        state=np.asarray(aligned['states']); original=np.asarray(baseline['states']); previous=np.asarray(preceding['states'])
        v=np.linalg.norm(state[:,1:,7:10]-original[:,1:,7:10],axis=2)
        omega=np.linalg.norm(state[:,1:,10:13]-original[:,1:,10:13],axis=2)
        report[lane]=dict(error_vs_failed_finest=engine.errors(baseline,aligned),
                          error_vs_preceding=engine.errors(preceding,aligned),
                          frame_velocity_rms_vs_finest=np.sqrt(np.mean(v*v,axis=1)).tolist(),
                          frame_omega_rms_vs_finest=np.sqrt(np.mean(omega*omega,axis=1)).tolist(),
                          bitwise_same_as_finest=result['states']==baseline['states'],
                          collision_updates=result['collision_updates'],boundary_work_J=result['boundary_work_J'],
                          step_s=result['step_s'],contact_residual_m_s=result['coulomb_residual_max_m_s'],
                          penetration_m=result['max_contact_penetration_m'],surface_excess_m=result['max_container_surface_excess_m'])
    with zipfile.ZipFile(DIRECTORY/'traces.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for lane,result in runs.items():archive.writestr(lane+'.json',json.dumps(result,separators=(',',':')))
    save(DIRECTORY/'summary.json',dict(provenance=json.loads((DIRECTORY/'provenance.json').read_text()),runs=report))
    print(json.dumps(report,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--lane');parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    if args.lane:run_lane(args.lane)
    if args.summarize:summarize()

if __name__=='__main__':main()
