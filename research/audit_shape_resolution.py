"""Independent raw-trace, geometry, source and qualification checks for follow-up."""
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

import numpy as np

ROOT=Path(__file__).parent/'random-shape-resolution'


def metrics(reference,candidate):
    assert reference['physical_setup_id']==candidate['physical_setup_id']
    np.testing.assert_array_equal(reference['times'],candidate['times'])
    np.testing.assert_allclose(reference['mass'],candidate['mass'],rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(reference['inertia'],candidate['inertia'],rtol=1e-5,atol=1e-7)
    r=np.asarray(reference['states']);c=np.asarray(candidate['states']);assert r.shape==c.shape
    d=c-r
    return {'rms_position_m':float(np.sqrt(np.mean(d[:,:,0]**2+d[:,:,1]**2))),
        'rms_velocity_m_s':float(np.sqrt(np.mean(d[:,:,3]**2+d[:,:,4]**2))),
        'rms_spin_rad_s':float(np.sqrt(np.mean(d[:,:,5]**2))),
        'max_position_m':float(np.max(np.sqrt(d[:,:,0]**2+d[:,:,1]**2))),
        'final_velocity_max_m_s':float(np.max(np.sqrt(d[-1,:,3]**2+d[-1,:,4]**2))),
        'final_spin_max_rad_s':float(np.max(np.abs(d[-1,:,5])))}


def normalized(error,budget):return max(error[k]/v for k,v in budget.items())


def moments(vertices):
    vertices=np.asarray(vertices); mass=0.; first=np.zeros(2); polar=0.
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        area=(a[0]*b[1]-a[1]*b[0])/2;mass+=area;first+=area*(a+b)/3
        polar+=area*(a@a+a@b+b@b)/6
    return mass,first,polar


def body_integrals(body):
    mass=0.;first=np.zeros(2);polar=0.
    for fixture in body['polygons']:
        a,f,j=moments(fixture['vertices']);d=fixture.get('density',1)
        mass+=d*a;first+=d*f;polar+=d*j
    return mass,first,polar


def audit():
    output=ROOT/'results';data=json.loads((output/'summary.json').read_text());plan=json.loads((ROOT/'plan.json').read_text())
    oldplan=json.loads((ROOT.parent/'random-shapes/plan.json').read_text())
    assert plan['reference_budget']==oldplan['reference_budget'] and plan['budget']==oldplan['budget']
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    for file,field in [('execution-source.zip','source_sha256'),('traces.zip','traces_sha256'),
        ('scenes.json','scenes_sha256'),('precision-source.json','precision_source_sha256')]:
        assert digest(output/file)==data[field]
    assert digest(ROOT/'plan.json')==data['plan_sha256']
    manifest=json.loads((output/'precision-source.json').read_text())
    with zipfile.ZipFile(output/'execution-source.zip') as source:
        assert source.read('research/random-shape-resolution/plan.json')==(ROOT/'plan.json').read_bytes()
        assert hashlib.sha256(source.read('research/build_precision_backend.py')).hexdigest()==manifest['transform_script_sha256']
        for name in ('runner.cpp','compat2.h'):
            assert hashlib.sha256(source.read('rigid_backend/'+name)).hexdigest()==manifest['inputs'][name]
    scenes=json.loads((output/'scenes.json').read_text());records={r['trace']:r for r in data['records']}
    expected=4*(5+4+1)+2*(1+7)+2+2
    assert len(records)==expected==60
    with zipfile.ZipFile(output/'traces.zip') as z:
        assert set(z.namelist())=={name+'.json' for name in records}
        traces={}
        for name,record in records.items():
            packet=json.loads(z.read(name+'.json'));assert packet['record']==record
            result=packet['result'];traces[name]=result;scene=scenes[record['scene']]
            state=np.asarray(result['states']);assert np.isfinite(state).all()
            assert np.isfinite(result['kinematic_states']).all()
            assert len(state)==round(scene['duration']*120)+1
            assert len(record['cost_samples_s'])==record['repeats']
            assert record['median_s']==statistics.median(record['cost_samples_s'])
            assert record['cost_samples_s'][-1]==result['engine_and_controller_s']
            assert abs(result['engine_and_controller_s']-result['step_s']-result['controller_s'])<1e-12
            model=result['numerical_model'];assert model['scalar_precision']==('float64' if record['double'] else 'float32')
            if record['double']:assert model['precision_source_sha256']==data['precision_source_sha256']
            for index,body in enumerate(b for b in scene['bodies'] if b.get('type','dynamic')=='dynamic'):
                mass,first,polar=body_integrals(body)
                np.testing.assert_allclose(result['mass'][index],mass,rtol=2e-6,atol=1e-7)
                np.testing.assert_allclose(result['inertia'][index],polar-first@first/mass,rtol=2e-5,atol=1e-7)
            if record['policy'] is not None:
                assert record['policy']==plan['adaptive_policy']
                assert sum(result['level_frames'])==len(state)-1
                assert all(x in (0,1,2,3) for x in result['selected_levels'])
                assert len(set(result['selected_levels']))>1
        for name,selection in data['concave_selections'].items():
            refinements=[]
            for a,b in plan['concave_edges']:
                key=lambda x:f'{name}__reference_p{x[0]}_s{x[1]}'
                error=metrics(traces[key(b)],traces[key(a)])
                refinements.append({'from':a,'to':b,'errors':error,'normalized_error':normalized(error,plan['reference_budget'])})
            assert selection['refinements']==refinements
            qualified=all(r['normalized_error']<=1 for r in refinements)
            if not qualified:
                assert selection['status']=='unqualified_reference' and selection['choice'] is None
                assert selection['comparisons']==[];continue
            ref=traces[name+'__reference_p512_s128'];comparisons=[]
            for label in [f'candidate_p{p}_s{s}' for p,s in plan['concave_candidates']]+['adaptive']:
                key=name+'__'+label;error=metrics(ref,traces[key])
                comparisons.append({'candidate':label,'errors':error,'normalized_error':normalized(error,plan['budget']),
                                    'cost_s':records[key]['median_s']})
            assert selection['comparisons']==comparisons
            passed=[c for c in comparisons if c['normalized_error']<=1]
            best=min(passed,key=lambda c:c['cost_s']) if passed else None
            assert selection['choice']==(best['candidate'] if best else None)
            assert selection['status']==('qualified' if best else 'no_candidate_within_budget')
        for name,q in data['packed_qualification'].items():
            scene=scenes[name];original=scenes[name.removesuffix('_exact_partition')]
            # Every original convex core is contained in a new convex core.
            # Equal total area and preserved materials then certify the same union.
            for old,new in zip(original['bodies'][1:],scene['bodies'][1:]):
                for a,b in zip(body_integrals(old),body_integrals(new)):
                    np.testing.assert_allclose(a,b,atol=1e-12)
                for fixture in old['polygons']:
                    contained=False
                    for target in new['polygons']:
                        if any(fixture.get(k,d)!=target.get(k,d) for k,d in [('density',1),('friction',.3),('restitution',0),('rolling',0)]):continue
                        vertices=np.asarray(target['vertices']);edges=np.roll(vertices,-1,axis=0)-vertices
                        p=np.asarray(fixture['vertices']);relative=p[None,:,:]-vertices[:,None,:]
                        cross=edges[:,0,None]*relative[:,:,1]-edges[:,1,None]*relative[:,:,0]
                        if np.min(cross)>=-1e-12:contained=True;break
                    assert contained
            refinement=[]
            key=lambda x:f'{name}__precision_p{x[0]}_s{x[1]}_position{x[2]}'
            for a,b in plan['packed_edges']:refinement.append({'from':a,'to':b,'errors':metrics(traces[key(b)],traces[key(a)])})
            assert q['refinements']==refinement
            assert q['qualified']==all(normalized(e['errors'],plan['reference_budget'])<=1 for e in refinement)
            for record in (r for r in records.values() if r['scene']==name):
                result=traces[record['trace']];assert result['numerical_model']['analytic_kinematics']
                assert result['numerical_model']['position_iterations']==record['position']
                times=np.asarray(result['times']);x=np.asarray(result['kinematic_states'])[:,0,0]
                expected=.6*times
                expected=np.where(times>.5,.3-.6*(times-.5),expected)
                expected=np.where(times>1,.6*(times-1),expected)
                np.testing.assert_allclose(x,expected,atol=1e-12)
        assert all(c['accepted'] and c['trace'] in traces for c in data['geometry_rechecks'])
        for item in data['precision_controls']:
            result=traces[item['trace']];error=abs(result['states'][-1][0][4]+9.81)
            assert error==item['analytic_velocity_error_m_s']
            if item['precision']=='float64':assert error<=plan['precision_control']['float64_velocity_tolerance_m_s']
        sensitivity=data['sensitivity'];a=traces[sensitivity['base']];b=traces[sensitivity['perturbed']]
        assert a['physical_setup_id']!=b['physical_setup_id']
        d=np.asarray(a['states'])-np.asarray(b['states'])
        measured={'rms_position_m':float(np.sqrt(np.mean(d[:,:,0]**2+d[:,:,1]**2))),
            'rms_velocity_m_s':float(np.sqrt(np.mean(d[:,:,3]**2+d[:,:,4]**2))),
            'rms_spin_rad_s':float(np.sqrt(np.mean(d[:,:,5]**2))),
            'initial_max_position_difference_m':float(np.max(np.sqrt(d[0,:,0]**2+d[0,:,1]**2)))}
        assert measured==sensitivity['errors']
    print(f'Audited 60 follow-up histories; {sum(q["status"]=="qualified" for q in data["concave_selections"].values())}/4 '
          f'concave references/selectors qualified, {sum(q["qualified"] for q in data["packed_qualification"].values())}/2 packed; '
          'geometry fixes, exact partition unions, full-precision provenance, boundary paths and original budgets verified.')


if __name__=='__main__':audit()
