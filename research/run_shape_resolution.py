"""Fresh validation of geometry fixes, concave references and packed controls."""
import copy
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import zipfile

import numpy as np

from fidelity import select
from rigid_engine import run, BINARIES
from research.random_shapes import scenes
from research.convex_partition import scene_partition
from research.rigid_scenes import body, rectangle, make_scene
from research.run_rigid_study import errors, normalized_error, diagnostics

ROOT=Path(__file__).parent/'random-shape-resolution'
DOUBLE=Path(__file__).resolve().parents[1]/'build/rigid_double/rigid_runner'


def state_errors(a,b):
    x=np.asarray(a['states']); y=np.asarray(b['states']); d=x-y
    return {'rms_position_m':float(np.sqrt(np.mean(np.sum(d[:,:,:2]**2,axis=2)))),
        'rms_velocity_m_s':float(np.sqrt(np.mean(np.sum(d[:,:,3:5]**2,axis=2)))),
        'rms_spin_rad_s':float(np.sqrt(np.mean(d[:,:,5]**2))),
        'initial_max_position_difference_m':float(np.max(np.linalg.norm(d[0,:,:2],axis=1)))}


def study():
    plan=json.loads((ROOT/'plan.json').read_text());output=ROOT/'results';output.mkdir(exist_ok=True)
    checkpoint=output/'checkpoints';checkpoint.mkdir(exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    files=['fidelity.py','rigid_engine.py','rigid_backend/runner.cpp','rigid_backend/compat2.h',
        'research/build_precision_backend.py','research/convex_partition.py','research/random_shapes.py',
        'research/rigid_scenes.py','research/container_scenes.py','research/run_rigid_study.py',
        'research/run_shape_resolution.py','research/random-shape-resolution/plan.json']
    with zipfile.ZipFile(output/'execution-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for file in files: archive.writestr(file,subprocess.check_output(['git','show',f'{commit}:{file}']))
    precision=DOUBLE.parent/'precision-source.json'
    (output/'precision-source.json').write_bytes(precision.read_bytes())
    original=json.loads((ROOT.parent/'random-shapes/results/scenes.json').read_text())
    held=[s for s in scenes(plan['held_out_concave_seeds']) if 'concave_drop' in s['id']]
    registry={s['id']:s for s in original+held};records=[];selections={};packed={};geometry=[]

    def execute(scene,label,*,backend='block',primary=8,solver=32,position=3,policy=None,
                double=False,repeats=None,warmups=None):
        repeats=plan['repeats'] if repeats is None else repeats
        warmups=plan['warmups'] if warmups is None else warmups
        key=scene['id']+'__'+label;registry[scene['id']]=scene
        kwargs={'backend':backend,'primary_steps':primary,'substeps':solver,'position_iterations':position,
                'policy':policy,'binary':DOUBLE if double else None}
        for _ in range(warmups):run(scene,**kwargs)
        samples=[run(scene,**kwargs) for _ in range(repeats)]
        result=samples[-1]; times=[r['engine_and_controller_s'] for r in samples]
        record={'trace':key,'scene':scene['id'],'backend':backend,'primary':primary,'solver':solver,
            'position':position,'policy':policy,'double':double,'warmups':warmups,'repeats':repeats,
            'cost_samples_s':times,'median_s':statistics.median(times),'diagnostics':diagnostics(scene,result)}
        temporary=checkpoint/(key+'.tmp');temporary.write_text(json.dumps({'record':record,'result':result},separators=(',',':')))
        temporary.replace(checkpoint/(key+'.json'));records.append(record)
        print(key,record['median_s'],flush=True);return result,record

    for scene in [s for s in original+held if 'concave_drop' in s['id']]:
        refs={}
        for p,s in plan['concave_reference_modes']:refs[p,s]=execute(scene,f'reference_p{p}_s{s}',primary=p,solver=s)[0]
        candidates={};costs={}
        for p,s in plan['concave_candidates']:
            label=f'candidate_p{p}_s{s}';candidates[label],record=execute(scene,label,primary=p,solver=s)
            costs[label]=record['median_s']
        candidates['adaptive'],record=execute(scene,'adaptive',policy=plan['adaptive_policy']);costs['adaptive']=record['median_s']
        edges=[(tuple(a),tuple(b)) for a,b in plan['concave_edges']]
        selections[scene['id']]=select(refs,(512,128),edges,candidates,reference_budget=plan['reference_budget'],
            budget=plan['budget'],costs=costs)
    for scene in [s for s in original if 'mixed36' in s['id']]:
        result,record=execute(scene,'geometry_recheck',backend='temporal',primary=4,solver=16)
        geometry.append({'scene':scene['id'],'trace':record['trace'],'accepted':True,
            'original_rejection':'Rigid backend failed: Convex ordered polygon required'})
        merged=scene_partition(scene);merged['id']+='_exact_partition';merged['analytic_kinematics']=True
        runs={}
        for p,s,k in plan['packed_modes']:
            runs[p,s,k]=execute(merged,f'precision_p{p}_s{s}_position{k}',primary=p,solver=s,position=k,
                               double=True,repeats=1,warmups=0)[0]
        refinement=[{'from':a,'to':b,'errors':errors(runs[tuple(b)],runs[tuple(a)])} for a,b in plan['packed_edges']]
        packed[merged['id']]={'qualified':all(normalized_error(e['errors'],plan['reference_budget'])<=1 for e in refinement),
                             'refinements':refinement,'partition_provenance':merged['partition_provenance']}
    base=next(s for s in original if s['id']=='random_42_mixed36_shake')
    a,ra=execute(base,'sensitivity_base',primary=128,solver=64,repeats=1,warmups=0)
    perturbed=copy.deepcopy(base);perturbed['id']+='_micrometre_perturbation'
    perturbed['bodies'][1]['position'][0]+=plan['sensitivity_control']['perturbation_m']
    b,rb=execute(perturbed,'sensitivity_perturbed',primary=128,solver=64,repeats=1,warmups=0)
    sensitivity={'base':ra['trace'],'perturbed':rb['trace'],'errors':state_errors(a,b),
                 'scope':plan['sensitivity_control']['scope']}
    control=make_scene('freefall_precision_control',[body(rectangle(),(0,5))],duration=1)
    precision_records=[]
    for double in (False,True):
        result,record=execute(control,'float64' if double else 'float32',primary=512,solver=8,double=double,repeats=1,warmups=0)
        precision_records.append({'trace':record['trace'],'precision':result['numerical_model']['scalar_precision'],
            'analytic_velocity_error_m_s':abs(result['states'][-1][0][4]+9.81)})
    (output/'scenes.json').write_text(json.dumps(registry,indent=2)+'\n')
    with zipfile.ZipFile(output/'traces.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for record in records:archive.write(checkpoint/(record['trace']+'.json'),record['trace']+'.json')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    summary={'source_commit':commit,'scope':plan['scope'],'machine':platform.platform(),
        'binary_sha256':{**{k:digest(v) for k,v in BINARIES.items()},'float64_diagnostic':digest(DOUBLE)},
        'plan_sha256':digest(ROOT/'plan.json'),'source_sha256':digest(output/'execution-source.zip'),
        'traces_sha256':digest(output/'traces.zip'),'scenes_sha256':digest(output/'scenes.json'),
        'precision_source_sha256':digest(output/'precision-source.json'),'records':records,
        'concave_selections':selections,'packed_qualification':packed,'geometry_rechecks':geometry,
        'sensitivity':sensitivity,'precision_controls':precision_records}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':study()
