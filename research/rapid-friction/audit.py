"""Independent archived-state geometry, inertia, accuracy and timing audit."""
import copy
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from research.audit_hull_search_completion import trajectory_metrics,trajectory_error

HERE=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text())
def sample(result):
    result=copy.deepcopy(result);t=np.array(result['times']);desired=np.arange(13)*.01
    index=np.searchsorted(t,desired);index=np.minimum(index,len(t)-1)
    # Permit only representational rounding of a matching output timestamp.
    for i,v in enumerate(desired):
        candidates=[k for k in (index[i],index[i]-1) if k>=0]
        index[i]=min(candidates,key=lambda k:abs(t[k]-v))
    assert np.max(abs(t[index]-desired))<1e-12
    result['times']=desired.tolist();result['states']=np.array(result['states'])[index].tolist();return result

def planar_integrals(body):
    mass=0.;first=np.zeros(2);polar=0.
    for p in body.get('polygons',[]):
        v=np.asarray(p['vertices']);w=np.roll(v,-1,axis=0);cross=v[:,0]*w[:,1]-v[:,1]*w[:,0];density=p.get('density',1.)
        mass+=density*np.sum(cross)/2;first+=density*np.sum((v+w)*cross[:,None],axis=0)/6
        polar+=density*np.sum(cross*(np.sum(v*v,axis=1)+np.sum(v*w,axis=1)+np.sum(w*w,axis=1)))/12
    for c in body.get('circles',[]):
        m=c.get('density',1)*np.pi*c['radius']**2;center=np.array(c.get('center',[0,0]));mass+=m;first+=m*center;polar+=m*(c['radius']**2/2+center@center)
    center=first/mass;return mass,center,polar-mass*(center@center)

def planar_metrics(scene,result):
    x=np.asarray(result['states']);boundary=np.asarray(result['kinematic_states'])[:,0];times=np.asarray(result['times'])
    assert np.all(np.isfinite(x)) and x.shape[2]==6
    half=scene['container_half_extents_m'][0];E=np.zeros(len(x));excess=-np.inf
    for i,body in enumerate(scene['bodies'][1:]):
        mass,com,inertia=planar_integrals(body)
        assert np.isclose(result['mass'][i],mass,rtol=1e-12,atol=1e-12)
        assert np.isclose(result['inertia'][i],inertia,rtol=1e-12,atol=1e-12)
        s=x[:,i];E+=mass*np.sum(s[:,3:5]**2,axis=1)/2+inertia*s[:,5]**2/2-mass*(s[:,:2]@np.asarray(scene['gravity']))
        for state,wall in zip(s,boundary):
            a=state[2]-wall[2];R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            a=-wall[2];Rc=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            center=Rc@(state[:2]-wall[:2])
            for polygon in body.get('polygons',[]):
                points=(np.array(polygon['vertices'])-com)@R.T+center
                excess=max(excess,float(np.max(abs(points))+scene['collision_skin_m']-half))
            for circle in body.get('circles',[]):
                p=center+R@(np.array(circle.get('center',[0,0]))-com)
                excess=max(excess,float(np.max(abs(p))+circle['radius']-half))
    return {'container_surface_excess_m':excess,'energy_change_minus_boundary_work_J':float(E[-1]-E[0]-result['boundary_work_J'])}

def physical(entry,result):
    if entry['dimension']==3:
        metrics=trajectory_metrics(entry['scene'],result)
        metrics.update(contact_residual_m_s=result['coulomb_residual_max_m_s'],position_residual_m_s=result['translation_split_residual_max_m_s'])
        limits={'quaternion_norm_error':1e-12,'energy_change_minus_boundary_work_J':1.,'container_surface_excess_m':.002,'contact_residual_m_s':1e-8,'position_residual_m_s':1e-8}
    else:
        metrics=planar_metrics(entry['scene'],result);limits={'energy_change_minus_boundary_work_J':1.,'container_surface_excess_m':.01}
    assert all(np.isfinite(metrics[k]) for k in limits)
    return all(metrics[k]<=v for k,v in limits.items())

def errors(dimension,a,b):
    a,b=sample(a),sample(b)
    if dimension==3:return trajectory_error(a,b)
    assert a['physical_setup_id']==b['physical_setup_id']
    assert a['mass']==b['mass'] and a['inertia']==b['inertia']
    x=np.array(a['states']);y=np.array(b['states']);d=x-y
    return {'position_m':float(np.sqrt(np.mean(d[:,:,0]**2+d[:,:,1]**2))),
            'velocity_m_s':float(np.sqrt(np.mean(d[:,:,3]**2+d[:,:,4]**2))),
            'omega_rad_s':float(np.sqrt(np.mean(d[:,:,5]**2))),
            'orientation_rad':float(np.sqrt(np.mean(np.arctan2(np.sin(d[:,:,2]),np.cos(d[:,:,2]))**2)))}

def audit():
    gates=load(HERE/'plan.json');report={'record_count':0,'qualified_benchmarks':{},'directories':{}}
    for dirname in ['results','results-spatial','results-planar-resolution','results-planar-shake','results-planar-tight','results-planar-optimized','results-large-irregular','results-planar-discovery']:
        directory=HERE/dirname
        if not (directory/'summary.json').exists():continue
        scenes=load(directory/'scenes.json') if (directory/'scenes.json').exists() else load(HERE/'results/scenes.json')
        summary=load(directory/'summary.json');count=0
        for path in directory.rglob('*.json'):
            record=load(path)
            if not isinstance(record,dict) or 'setting' not in record or 'complete' not in record:continue
            count+=1
            if record['complete']:
                entry=scenes[path.parent.name];passed=physical(entry,record['result'])
                if dirname in ('results-planar-shake','results-planar-tight','results-planar-optimized'):passed &= record['result']['friction_impulse_abs_kg_m_s']>0
                assert passed==record['physical']['passed'],str(path)
        for name,item in summary.items():
            entry=scenes[name];budget=gates['trajectory_budgets'][str(entry['dimension'])];folder=directory/name
            if 'reference_archive' in item:
                prior=load(HERE/item['reference_archive']/'summary.json')[name]
                assert prior['reference_qualified'] and all(e['passed'] for e in prior['phases'][0]['edges'][-2:])
                reference=load(HERE/item['reference_archive']/item['reference_record'])['result']
                assert item['reference_qualified']
                for candidate in item['candidates']:
                    setting=candidate['setting'];record=load(folder/f"candidate_{setting['primary_steps']}_{setting['substeps']}.json")
                    error=errors(2,reference,record['result'])
                    passed=record['physical']['passed'] and all(error[k]<=v for k,v in budget.items())
                    assert bool(passed)==candidate['passed']
                    for k in error:assert np.isclose(error[k],candidate['errors'][k],rtol=1e-10,atol=1e-12)
            elif 'phases' in item:
                for phase in item['phases']:
                    refs=[load(folder/(phase['phase']['id']+f'_reference_{i}.json')) for i in range(len(phase['edges'])+1)]
                    for i,edge in enumerate(phase['edges']):
                        a,b=refs[i:i+2];error=errors(entry['dimension'],a['result'],b['result']) if a['complete'] and b['complete'] else None
                        passed=error is not None and a['physical']['passed'] and b['physical']['passed'] and all(error[k]<=v/4 for k,v in budget.items())
                        assert passed==edge['passed']
                    assert phase['qualified']==all(e['passed'] for e in phase['edges'][-2:])
                assert item['reference_qualified']==any(p['qualified'] for p in item['phases'])
                qualified_phase=next((p for p in item['phases'] if p['qualified']),None)
                reference=load(folder/(qualified_phase['phase']['id']+f"_reference_{len(qualified_phase['edges'])}.json"))['result'] if qualified_phase else None
            else:
                for edge in item['edges']:
                    a=load(folder/f"reference_{edge['left']}.json");b=load(folder/f"reference_{edge['right']}.json")
                    error=errors(entry['dimension'],a['result'],b['result']) if a['complete'] and b['complete'] else None
                    passed=error is not None and a['physical']['passed'] and b['physical']['passed'] and all(error[k]<=v/4 for k,v in budget.items())
                    assert passed==edge['passed']
                    if error is not None:
                        for k in error:assert np.isclose(error[k],edge['errors'][k],rtol=1e-10,atol=1e-12)
                assert item['reference_qualified']==all(e['passed'] for e in item['edges'][-2:])
                reference=load(folder/f"reference_{item['reference_levels_executed']-1}.json")['result'] if item['reference_qualified'] else None
            bench=item.get('benchmark')
            if bench is None:
                assert dirname=='results-planar-discovery' or not item['reference_qualified']
                continue
            assert item['reference_qualified']
            sampled_states={k:[] for k in ('reference','candidate')};timings={k:[] for k in sampled_states};allpass=True
            for warmup_path in folder.glob('warmup_*.json'):
                warmup=load(warmup_path);allpass &= warmup['complete'] and warmup['physical']['passed']
                if warmup['complete']:
                    error=errors(entry['dimension'],reference,warmup['result']);allpass &= all(error[k]<=v for k,v in budget.items())
            for recordpath in sorted(folder.glob('timing_*.json')):
                label=recordpath.stem.split('_')[-1];record=load(recordpath);allpass &= record['complete'] and record['physical']['passed']
                if record['complete']:
                    error=errors(entry['dimension'],reference,record['result']);allpass &= all(error[k]<=v for k,v in budget.items())
                    sampled_states[label].append(np.array(record['result']['states'],dtype=np.float64).tobytes());timings[label].append(record['result']['step_s'])
            deterministic=all(len(set(v))==1 for v in sampled_states.values())
            assert bench['qualified']==bool(allpass and deterministic)
            assert timings==bench['samples_s'] and all(len(v)==3 for v in timings.values())
            if bench['qualified']:
                medians={k:statistics.median(v) for k,v in timings.items()};ratio=medians['reference']/medians['candidate']
                assert medians==bench['median_s'] and ratio==bench['reference_over_candidate']
                report['qualified_benchmarks'][name]={'median_s':medians,'ratio':ratio,'record_directory':dirname}
                if 'adapter_process_samples_s' in bench:
                    process={k:[load(folder/f'timing_{i}_{k}.json')['result']['wall_time_s'] for i in range(3)] for k in timings}
                    assert process==bench['adapter_process_samples_s']
                    process_medians={k:statistics.median(v) for k,v in process.items()}
                    assert process_medians==bench['adapter_process_median_s']
                    assert process_medians['reference']/process_medians['candidate']==bench['adapter_process_ratio']
                    report['qualified_benchmarks'][name]['adapter_process_median_s']=process_medians
        report['directories'][dirname]={'records':count,'qualified_references':sum(i['reference_qualified'] for i in summary.values())};report['record_count']+=count
    report['passed']=True
    (HERE/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':audit()
