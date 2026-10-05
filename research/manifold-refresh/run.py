"""Compile isolated geometry-cache prototypes, then execute the frozen protocol."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('rapid_frozen',ROOT/'research/rapid-friction/run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def main():
    plan=json.loads((HERE/'plan.json').read_text());out=HERE/'results';out.mkdir(exist_ok=False)
    binarydir=Path('/home/viktor/.cache/physics-manifold-refresh-01');binarydir.mkdir(exist_ok=False)
    sourceguards=json.loads((HERE/'baseline-source.json').read_text())
    for p,h in sourceguards.items():assert base.digest(ROOT/p)==h
    libs=[ROOT/'build/spatial/_deps/bullet-build/src'/p for p in ['BulletDynamics/libBulletDynamics.a','BulletCollision/libBulletCollision.a','LinearMath/libLinearMath.a']]
    include=Path('/home/viktor/.cache/physics-relation-20261005/deps/_deps')
    binaries={};commands=[]
    for policy in plan['policies']:
        target=binarydir/policy
        command=['c++','-std=c++17','-O2','-ffp-contract=off','-DBT_USE_DOUBLE_PRECISION','-DSPATIAL_LAPACK_RECOVERY=1','-I'+str(ROOT/'spatial_backend'),'-I'+str(include/'bullet-src/src'),'-I'+str(include/'json-src/single_include'),str(HERE/(policy+'_runner.cpp')),*map(str,libs),'/lib/x86_64-linux-gnu/liblapack.so.3','/lib/x86_64-linux-gnu/libblas.so.3','-o',str(target)]
        result=subprocess.run(command,capture_output=True,text=True);base.atomic(out/(policy+'-compile.json'),{'command':command,'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        commands.append(command)
        if result.returncode:raise SystemExit(result.returncode)
        binaries[policy]=target
    guards={str(ROOT/p):h for p,h in sourceguards.items()}
    guards.update({str(p):base.digest(p) for p in [HERE/'run.py',HERE/'plan.json',HERE/'cold_runner.cpp',HERE/'tight_runner.cpp',*binaries.values(),*libs]})
    base.atomic(out/'provenance.json',{'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'guards':guards,'commands':commands})
    entries=json.loads((ROOT/'research/rapid-friction/results-spatial/scenes.json').read_text());gates=json.loads((ROOT/'research/rapid-friction/plan.json').read_text());summary={}
    for policy in plan['policies']:
        summary[policy]={}
        for name in plan['scenes']:
            entry=entries[name];refs=[];edges=[]
            for i,setting in enumerate(plan['reference_levels']):
                assert all(base.digest(p)==h for p,h in guards.items())
                dest=out/policy/name/f'reference_{i}.json';dest.parent.mkdir(parents=True,exist_ok=True)
                start=time.perf_counter()
                try:
                    result=base.spatial_run(entry['scene'],solver='coulomb',iterations=4096,kinematic_contact_phase='start',position_stabilization='split_translation_combined',contact_point_policy='shared',contact_tolerance_m_s=1e-8,contact_slop_m=1e-9,contact_recovery=True,early_component_recovery=True,binary=binaries[policy],rejected_contact_path=str(dest.with_suffix('.rejection.json')),**setting)
                    record={'complete':True,'result':result,'physical':base.physical(entry,result,gates),'setting':setting,'process_elapsed_s':time.perf_counter()-start}
                except subprocess.CalledProcessError as e:record={'complete':False,'stderr':e.stderr,'exit':e.returncode,'setting':setting,'process_elapsed_s':time.perf_counter()-start}
                base.atomic(dest,record);refs.append(record)
                if i:
                    a,b=refs[-2:];error=base.errors(3,a['result'],b['result']) if a['complete'] and b['complete'] else None
                    passed=bool(error is not None and a['physical']['passed'] and b['physical']['passed'] and all(error[k]<=v/4 for k,v in plan['trajectory_budget'].items()))
                    edges.append({'passed':passed,'errors':error})
                print(policy,name,i,'complete',record['complete'],'edge',edges[-1] if edges else None,flush=True)
                assert all(base.digest(p)==h for p,h in guards.items())
            summary[policy][name]={'qualified':all(e['passed'] for e in edges),'edges':edges,'histories':sum(r['complete'] for r in refs)};base.atomic(out/'summary.json',summary)
    base.atomic(out/'final.json',{'complete':True,'guards_unchanged':True,'summary':summary})

if __name__=='__main__':main()
