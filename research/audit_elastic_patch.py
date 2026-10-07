"""Independent bookkeeping audit of archived elastic sphere-plane histories."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np


def audit(directory=Path('research/elastic-patch')):
    summary=json.loads((directory/'summary.json').read_text()); checked=0
    with zipfile.ZipFile(directory/'execution-source.zip') as sources:
        for name,digest in summary['source_sha256'].items():
            assert hashlib.sha256(sources.read(name)).hexdigest()==digest,name
        plan_names=[p for p in summary['source_sha256'] if p.endswith('/plan.json')]
        assert len(plan_names)==1
        plan=json.loads(sources.read(plan_names[0]))
    cases={c['name']:c for c in plan['cases']}; archive_names=[]; gates=plan['gates']
    with zipfile.ZipFile(directory/'traces.zip') as archive:
        archive_names=archive.namelist()
        for row in summary['cases']:
            case=cases[row['name']]; m=case['material']; sim=case['simulation']
            mass=m['mass']; R=m['radius']; inertia=.4*mass*R**2
            gravity=np.asarray(sim.get('gravity',[0.,0.,0.])); a=m['effective_length']
            K=np.array([m['tangent_stiffness'],m['tangent_stiffness'],m['twist_stiffness']/a**2])
            planes=sim.get('planes',[dict(normal=[0,0,1],offset=0,name='floor')])
            retained=[]
            for run in row['runs']:
                name=row['name']+'/'+run['level']+'.npz'
                if run['status']=='rejected':
                    assert name not in archive_names and run['reason']; retained.append(None); continue
                with np.load(io.BytesIO(archive.read(name))) as trace:
                    times=trace['times']; states=trace['states']; strain=trace['strain']
                    assert np.all(np.diff(times)>0)
                    assert times[0]==0 and abs(times[-1]-sim['duration'])<1e-14
                    for key in trace.files: assert np.all(np.isfinite(trace[key])),name+':'+key
                    kinetic=.5*mass*np.sum(states[:,3:6]**2,axis=1)+.5*inertia*np.sum(states[:,6:9]**2,axis=1)
                    potential=-mass*states[:,:3]@gravity
                    stored=np.zeros(len(times))
                    for j,p in enumerate(planes):
                        n=np.asarray(p['normal']); delta=np.maximum(0,R-(states[:,:3]@n-p['offset']))
                        f=np.where(delta>0,(delta/R)**m['compression_exponent'],0.)
                        h=strain[:,3*j:3*j+3]
                        stored+=.5*m['normal_stiffness']*delta**2+.5*f*np.sum(h*K*h,axis=1)
                        scaled_force=f[:,None]*h*K
                        sampled_excess=np.linalg.norm(scaled_force,axis=1)-m['friction']*m['normal_stiffness']*delta
                        assert np.max(sampled_excess)<=run['metrics']['max_yield_excess_N']+1e-8
                    np.testing.assert_allclose(trace['kinetic_J'],kinetic,rtol=1e-12,atol=1e-10)
                    np.testing.assert_allclose(trace['potential_J'],potential,rtol=1e-12,atol=1e-10)
                    np.testing.assert_allclose(trace['stored_J'],stored,rtol=1e-12,atol=1e-10)
                    total=kinetic+potential+stored+trace['dissipated_J']
                    np.testing.assert_allclose(trace['energy_residual_J'],total-total[0],rtol=1e-10,atol=1e-10)
                    np.testing.assert_allclose(mass*(states[:,3:6]-states[0,3:6]),
                                               trace['linear_impulse_N_s']+mass*times[:,None]*gravity,
                                               rtol=1e-9,atol=1e-6)
                    if len(planes)==1:
                        lever=-R*np.asarray(planes[0]['normal'])
                        expected=np.cross(lever,trace['linear_impulse_N_s'])+trace['couple_impulse_N_m_s']
                        np.testing.assert_allclose(inertia*(states[:,6:9]-states[0,6:9]),expected,rtol=1e-9,atol=1e-7)
                    assert abs(float(np.max(abs(trace['energy_residual_J'])))-run['metrics']['max_energy_residual_J'])<1e-10
                    np.testing.assert_allclose(states[-1,3:6],run['metrics']['final_velocity_m_s'],rtol=0,atol=1e-12)
                    np.testing.assert_allclose(states[-1,6:9],run['metrics']['final_omega_rad_s'],rtol=0,atol=1e-12)
                    retained.append({key:trace[key].copy() for key in trace.files})
                    checked+=1
                assert run['elapsed_s']>=0
            recomputed={}
            if all(r is not None for r in retained):
                recomputed['energy']=all(np.max(abs(r['energy_residual_J']))<=
                    gates['energy_absolute_J']+gates['energy_relative']*abs(
                    r['kinetic_J'][0]+r['potential_J'][0]+r['stored_J'][0]) for r in retained)
                # Internal accepted-step peak is source-recorded; the archived
                # sampled forces were independently checked against that peak.
                recomputed['yield']=all(run['metrics']['max_yield_excess_N']<=gates['yield_excess_N'] for run in row['runs'])
                recomputed['nonnegative_dissipation']=all(np.min(r['dissipated_J'])>=-gates['energy_absolute_J'] and
                    np.min(np.diff(r['dissipated_J']))>=-gates['energy_absolute_J'] for r in retained)
                edges=[]
                for a,b in zip(retained,retained[1:]):
                    np.testing.assert_array_equal(a['times'],b['times'])
                    difference=a['states']-b['states']
                    edge=dict(position_m=float(np.max(np.linalg.norm(difference[:,:3],axis=1))),
                              velocity_m_s=float(np.max(np.linalg.norm(difference[:,3:6],axis=1))),
                              omega_rad_s=float(np.max(np.linalg.norm(difference[:,6:9],axis=1))))
                    edges.append(edge)
                for recomputed_edge,claimed_edge in zip(edges,row['refinement_edges']):
                    for key in recomputed_edge: assert abs(recomputed_edge[key]-claimed_edge[key])<1e-12
                recomputed['refinement']=all(e['position_m']<=gates['position_refinement_m'] and
                    e['velocity_m_s']<=gates['velocity_refinement_m_s'] and
                    e['omega_rad_s']<=gates['omega_refinement_rad_s'] for e in edges)
                fine=retained[-1]; met=row['runs'][-1]['metrics']
                if 'expected_velocity' in sim:
                    recomputed['analytic_velocity']=np.linalg.norm(fine['states'][-1,3:6]-sim['expected_velocity'])<=gates['analytic_velocity_m_s']
                if 'expected_omega' in sim:
                    recomputed['analytic_omega']=np.linalg.norm(fine['states'][-1,6:9]-sim['expected_omega'])<=gates['analytic_omega_rad_s']
                if 'expected_dissipation' in sim:
                    recomputed['analytic_dissipation']=abs(fine['dissipated_J'][-1]-sim['expected_dissipation'])<=1e-5
                if 'require_spin_reversal' in sim:
                    recomputed['spin_reversal']=fine['states'][-1,8]*sim['omega'][2]<0
                if 'minimum_bounces' in sim:
                    lifts=[e for e in met['events'] if e['kind']=='lift_off']
                    assert len(lifts)==met['bounces']
                    recomputed['bounces']=len(lifts)>=sim['minimum_bounces']
                    recomputed['alternating_bounce_velocities']=all(abs(e['velocity_m_s'][2]-(-1)**i)<1e-5 and
                        abs(e['omega_rad_s'][2]+10*(-1)**i)<1e-4 for i,e in enumerate(lifts))
                loss=sum(e['separation_loss_J'] for e in met['events'])
                assert abs(loss-met['separation_loss_J'])<1e-12
                recomputed['separation_loss']=loss<=gates['separation_loss_J']
            recomputed={key:bool(value) for key,value in recomputed.items()}
            assert row['checks']==recomputed,(row['name'],row['checks'],recomputed)
            assert row['qualified']==(all(recomputed.values()) and len(recomputed)>0)
    assert checked==summary['completed_histories']==len(archive_names)
    assert summary['qualified_cases']==sum(r['qualified'] for r in summary['cases'])
    rejected=sum(r['status']=='rejected' for c in summary['cases'] for r in c['runs'])
    assert rejected==summary['rejected_attempts']
    print(f'Elastic source/bookkeeping audit: {checked} histories, {rejected} retained rejections, {summary["qualified_cases"]}/{len(summary["cases"])} qualified cases')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--directory',default='research/elastic-patch')
    audit(Path(parser.parse_args().directory))
