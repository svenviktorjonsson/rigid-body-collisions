"""Prospective full-world rapid group-motion qualification and repeated timing."""
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rigid_engine import run as planar_run
from spatial_engine import run as spatial_run, errors as spatial_errors
from research.container_scenes import container as planar_container, ball
from research.random_shapes import generate
from research.rigid_scenes import body
from research.spatial_scenes import container as spatial_container
from research.spatial_metrics import diagnostics as spatial_diagnostics

HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def scenes():
    result={}
    for kind,side in [('disks',5),('mixed_polygons',3)]:
        half=(side-1)*.24/2+.13
        wall=planar_container(half,half,velocity=(20,0),omega=5,friction=.4)
        wall['velocity_schedule']=[{'time_s':.04,'velocity':[-20,0],'omega':-5},{'time_s':.08,'velocity':[20,0],'omega':5}]
        rng=np.random.default_rng(7301);bodies=[wall]
        for i in range(side*side):
            pos=((i%side-(side-1)/2)*.24,(i//side-(side-1)/2)*.24)
            if kind=='disks':b=ball(pos,friction=.4)
            else:
                pieces,_=generate(rng,concave=i%3==0,radius=.1,friction=.4)
                b=body(pieces,pos,angle=float(rng.uniform(-np.pi,np.pi)))
            bodies.append(b)
        scene={'id':'2d_'+kind,'duration':.12,'gravity':[0,-9.81],'collision_skin_m':.01,'analytic_kinematics':True,'bodies':bodies,'container_half_extents_m':[half,half]}
        result[scene['id']]={'dimension':2,'scene':scene}
    for kind,side,spin in [('sphere',3,0),('box',2,10),('hull',2,0)]:
        scene,_=spatial_container(side=side,speed=20,shake=True,spin=spin,shape=kind,seed=42,duration=.12)
        result['3d_'+kind]={'dimension':3,'scene':scene}
    return result

def sampled(result):
    r=copy.deepcopy(result);times=np.asarray(r['times']);indices=np.round(np.arange(13)*.01/(times[1]-times[0])).astype(int)
    assert np.allclose(times[indices],np.arange(13)*.01,atol=1e-12,rtol=0)
    r['times']=times[indices].tolist();r['states']=np.asarray(r['states'])[indices].tolist()
    return r

def errors(dimension,a,b):
    a,b=sampled(a),sampled(b)
    if dimension==3:return spatial_errors(a,b)
    assert a['physical_setup_id']==b['physical_setup_id'] and a['mass']==b['mass'] and a['inertia']==b['inertia']
    x,y=np.asarray(a['states']),np.asarray(b['states']);delta=x-y
    angle=np.arctan2(np.sin(delta[:,:,2]),np.cos(delta[:,:,2]))
    return {'position_m':float(np.sqrt(np.mean(np.sum(delta[:,:,:2]**2,axis=2)))),
            'velocity_m_s':float(np.sqrt(np.mean(np.sum(delta[:,:,3:5]**2,axis=2)))),
            'omega_rad_s':float(np.sqrt(np.mean(delta[:,:,5]**2))),
            'orientation_rad':float(np.sqrt(np.mean(angle**2)))}

def physical(entry,result,plan):
    scene=entry['scene'];dimension=entry['dimension']
    if dimension==3:
        metrics=spatial_diagnostics(scene,result,scene['container_interior_half_extents_m'][0])
        metrics['contact_residual_m_s']=result['coulomb_residual_max_m_s']
        metrics['position_residual_m_s']=result['translation_split_residual_max_m_s']
        limits={'quaternion_norm_error':1e-12,'energy_change_minus_boundary_work_J':1.,'container_surface_excess_m':.002,'contact_residual_m_s':1e-8,'position_residual_m_s':1e-8}
    else:
        state=np.asarray(result['states']);wall=np.asarray(result['kinematic_states'])[:,0];half=scene['container_half_extents_m'][0]
        excess=-np.inf
        for frame,box in zip(state,wall):
            angle=-box[2];rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            for s,authored in zip(frame,scene['bodies'][1:]):
                center=rotation@(s[:2]-box[:2])
                for circle in authored.get('circles',[]):excess=max(excess,float(np.max(abs(center))+circle['radius']-half))
                for polygon in authored.get('polygons',[]):
                    angle=s[2]-box[2];r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                    points=np.asarray(polygon['vertices'])@r.T+center
                    excess=max(excess,float(np.max(abs(points))+scene['collision_skin_m']-half))
        mass=np.asarray(result['mass']);inertia=np.asarray(result['inertia']);gravity=np.asarray(scene['gravity'])
        total=.5*np.sum(mass[None,:]*np.sum(state[:,:,3:5]**2,axis=2)+inertia[None,:]*state[:,:,5]**2,axis=1)-np.einsum('tij,j,i->t',state[:,:,:2],gravity,mass)
        metrics={'container_surface_excess_m':excess,'energy_change_minus_boundary_work_J':float(total[-1]-total[0]-result['boundary_work_J'])}
        limits={'container_surface_excess_m':.01,'energy_change_minus_boundary_work_J':1.}
    passed=bool(np.all(np.isfinite(result['states'])) and all(np.isfinite(metrics[k]) and metrics[k]<=v for k,v in limits.items()))
    return {'passed':passed,'metrics':metrics,'limits':limits}

def execute(entry,setting,output,plan):
    assert not output.exists(),output
    begun=time.perf_counter()
    try:
        if entry['dimension']==3:
            result=spatial_run(entry['scene'],solver='coulomb',iterations=4096,kinematic_contact_phase='start',position_stabilization='split_translation_combined',contact_point_policy='shared',contact_tolerance_m_s=1e-8,contact_slop_m=1e-9,contact_recovery=True,early_component_recovery=True,rejected_contact_path=str(output.with_suffix('.rejection.json')),**setting)
        else:
            planar_setting={k:v for k,v in setting.items() if k!='travel_fraction'}
            result=planar_run(entry['scene'],backend='block',substeps=32,position_iterations=12,binary=ROOT/'build/rigid_double_ledger/rigid_runner',**planar_setting)
        record={'complete':True,'result':result,'physical':physical(entry,result,plan),'process_elapsed_s':time.perf_counter()-begun,'setting':setting}
    except (subprocess.CalledProcessError,RuntimeError) as e:
        record={'complete':False,'error':str(e),'stderr':getattr(e,'stderr',None),'exit':getattr(e,'returncode',None),'process_elapsed_s':time.perf_counter()-begun,'setting':setting}
    atomic(output,record);return record

def main():
    plan=json.loads((HERE/'plan.json').read_text());out=HERE/'results';out.mkdir(exist_ok=False)
    assert os.environ.get('OMP_NUM_THREADS')=='1' and os.environ.get('OPENBLAS_NUM_THREADS')=='1'
    authored=scenes();atomic(out/'scenes.json',authored)
    paths=[ROOT/'build/spatial/spatial_runner',ROOT/'build/rigid_double_ledger/rigid_runner',ROOT/'build/rigid_double_ledger/precision-source.json',HERE/'run.py',HERE/'plan.json',ROOT/'spatial_engine.py',ROOT/'rigid_engine.py']
    for folder in ['spatial_backend','rigid_backend']:
        paths += [p for p in (ROOT/folder).glob('*') if p.is_file()]
    guards={str(p):digest(p) for p in paths}
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    provenance={'source':source,'guards':guards,'plan_sha256':digest(HERE/'plan.json'),'thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},'runtime':{}}
    for binary in paths[:2]:
        import re
        libs=re.findall(r'(/\S+)\s+\(',subprocess.check_output(['ldd',str(binary)],text=True))
        provenance['runtime'][str(binary)]={str(Path(p).resolve()):digest(Path(p).resolve()) for p in libs}
    atomic(out/'provenance.json',provenance);summary={}
    for name,entry in authored.items():
        refs=[];edges=[];budget=plan['trajectory_budgets'][str(entry['dimension'])];qualified=False
        for i,setting in enumerate(plan['reference_levels']):
            record=execute(entry,setting,out/name/f'reference_{i}.json',plan);refs.append(record)
            print(name,'reference',i,'complete',record['complete'],flush=True)
            if i:
                left,right=refs[-2:];error=errors(entry['dimension'],left['result'],right['result']) if left['complete'] and right['complete'] else None
                passed=bool(error is not None and left['physical']['passed'] and right['physical']['passed'] and all(error[k]<=v/4 for k,v in budget.items()))
                edges.append({'left':i-1,'right':i,'passed':passed,'errors':error});print('EDGE',name,edges[-1],flush=True)
            if i>=2 and all(e['passed'] for e in edges[-2:]):qualified=True;break
        item={'reference_qualified':qualified,'edges':edges,'reference_levels_executed':len(refs),'candidates':[],'benchmark':None};summary[name]=item
        if qualified:
            reference=refs[-1]['result'];passing=[]
            for i,setting in enumerate(plan['candidate_settings'][str(entry['dimension'])]):
                record=execute(entry,setting,out/name/f'candidate_{i}.json',plan)
                error=errors(entry['dimension'],reference,record['result']) if record['complete'] else None
                passed=bool(record['complete'] and record['physical']['passed'] and all(error[k]<=v for k,v in budget.items()))
                timing=record['result']['step_s'] if record['complete'] else None
                candidate={'index':i,'setting':setting,'passed':passed,'errors':error,'native_s':timing};item['candidates'].append(candidate)
                if passed:passing.append(candidate)
                print(name,'candidate',i,'passed',passed,flush=True)
            # Always include the coarsest of the qualifying three references.
            candidate={'index':'qualified_reference','setting':plan['reference_levels'][len(refs)-3],'passed':True,'errors':errors(entry['dimension'],reference,refs[-3]['result']),'native_s':refs[-3]['result']['step_s']}
            assert all(candidate['errors'][k]<=v for k,v in budget.items());passing.append(candidate)
            selected=min(passing,key=lambda c:c['native_s']);item['selected']=selected
            settings={'reference':plan['reference_levels'][len(refs)-1],'candidate':selected['setting']}
            warmups=[execute(entry,setting,out/name/f'warmup_{label}.json',plan) for label,setting in settings.items()]
            measured={'reference':[],'candidate':[]};measured_states={'reference':[],'candidate':[]};allpass=all(r['complete'] and r['physical']['passed'] for r in warmups)
            for repetition in range(plan['timing_repetitions']):
                order=['reference','candidate'] if repetition%2==0 else ['candidate','reference']
                for label in order:
                    record=execute(entry,settings[label],out/name/f'timing_{repetition}_{label}.json',plan)
                    passed=record['complete'] and record['physical']['passed']
                    if record['complete']:
                        error=errors(entry['dimension'],reference,record['result']);passed &= all(error[k]<=v for k,v in budget.items())
                        measured[label].append(record['result']['step_s']);measured_states[label].append(np.asarray(record['result']['states']).tobytes())
                    allpass &= passed
            deterministic=all(len(set(v))==1 for v in measured_states.values())
            benchmark={'qualified':bool(allpass and deterministic),'samples_s':measured,'states_bitwise_repeated':deterministic,'settings':settings}
            if benchmark['qualified']:
                benchmark['median_s']={k:statistics.median(v) for k,v in measured.items()};benchmark['reference_over_candidate']=benchmark['median_s']['reference']/benchmark['median_s']['candidate']
            item['benchmark']=benchmark;print('BENCHMARK',name,benchmark,flush=True)
        assert all(digest(p)==h for p,h in guards.items()),'Frozen source/binary changed'
        atomic(out/'summary.json',summary)
    atomic(out/'final.json',{'complete':True,'source_unchanged':True,'summary':summary})

if __name__=='__main__':main()
