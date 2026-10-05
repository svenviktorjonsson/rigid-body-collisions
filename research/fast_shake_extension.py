"""Prospective finer fast-shake references using the immutable diagnostic binary."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
from research.fast_shake_diagnostic import DIRECTORY,ROOT,BINARY,adapter,digest,save
from research.spatial_metrics import diagnostics

PLAN=DIRECTORY/'extension-plan.json'

def run_lane(lane):
    plan=json.loads(PLAN.read_text()); provenance=json.loads((DIRECTORY/'provenance.json').read_text())
    if digest(BINARY.read_bytes())!=provenance['binary_sha256']:raise RuntimeError('Frozen binary changed')
    if digest(Path('/tmp/fast-shake-diagnostic-adapter.py').read_bytes())!=provenance['source_hashes']['spatial_engine.py']:raise RuntimeError('Frozen adapter changed')
    authored=json.loads((DIRECTORY/'scene.json').read_text())['scene']
    common=json.loads((DIRECTORY/'plan.json').read_text())['common']
    start=time.perf_counter()
    try: result=adapter().run(authored,binary=BINARY,**common,**plan['runs'][lane])
    except subprocess.CalledProcessError as error: result=dict(rejected=error.stderr.strip(),exit_code=error.returncode)
    result['diagnostic_elapsed_s']=time.perf_counter()-start
    save(DIRECTORY/(lane+'.json'),result)
    print(lane,'rejected '+result['rejected'] if 'rejected' in result else f"updates={result['collision_updates']} native_s={result['step_s']:.3f}",flush=True)

def summarize():
    plan=json.loads(PLAN.read_text()); engine=adapter(); summaries={}; runs={}
    authored=json.loads((DIRECTORY/'scene.json').read_text())['scene']; half=authored['container_interior_half_extents_m'][0]
    for names in plan['references'].values():
        for lane in names:
            path=DIRECTORY/(lane+'.json')
            if path.exists(): runs[lane]=json.loads(path.read_text())
    for group,names in plan['references'].items():
        edges=[]; physical={}; qualified=True
        for lane in names:
            if lane not in runs or 'rejected' in runs[lane]: qualified=False;continue
            d=diagnostics(authored,runs[lane],half)
            physical[lane]=dict(metrics=d,passed=all(d[k]<=value for k,value in plan['physical_gates'].items()))
            qualified &= physical[lane]['passed']
        for left,right in zip(names[:-1],names[1:]):
            if left not in physical or right not in physical:
                edges.append(dict(left=left,right=right,passed=False,error=None));qualified=False;continue
            error=engine.errors(runs[left],runs[right]);passed=all(error[k]<=v/4 for k,v in plan['trajectory_budget'].items())
            qualified &= passed;edges.append(dict(left=left,right=right,passed=passed,error=error))
        summaries[group]=dict(qualified=qualified,edges=edges,physical=physical)
    save(DIRECTORY/'extension-summary.json',summaries)
    with zipfile.ZipFile(DIRECTORY/'extension-traces.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for lane,result in runs.items():archive.writestr(lane+'.json',json.dumps(result,separators=(',',':')))
    print(json.dumps({k:dict(qualified=v['qualified'],edges=v['edges']) for k,v in summaries.items()},indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lane');parser.add_argument('--summarize',action='store_true');args=parser.parse_args()
    if args.lane:run_lane(args.lane)
    if args.summarize:summarize()

if __name__=='__main__':main()
