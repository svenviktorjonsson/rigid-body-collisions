"""Independent momentum/energy, integration and data-partition controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gamma
from model import NormalTable,normal_reference,rolling_advance

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        result=json.loads((args.evidence/'results.json').read_text())
        source=json.loads((ROOT/'research/full-experimental-report/evidence-v3/results.json').read_text())
        tennis_controls=0;loo_controls=0
        for measured,fitted in zip(source['tennis']['datasets'],result['tennis']['datasets']):
            assert measured['specimen']==fitted['specimen']
            train=[p for p in measured['points'] if p['split']=='train']
            total_x_y=sum(p['apparent_omega_rad_s']*p['measured_effective_mu_r'] for p in train)
            total_x_x=sum(p['apparent_omega_rad_s']**2 for p in train)
            fitted_a=total_x_y/total_x_x
            assert math.isclose(fitted_a,fitted['estimated_effective_relaxation_s'],rel_tol=1e-14)
            assert fitted_a>=0 and len(train)==3
            for original,new in zip(measured['points'],fitted['records']):
                assert original['split']==new['split']
                assert original['apparent_omega_rad_s']==new['omega_rad_s']
                assert original['measured_effective_mu_r']==new['observed_mu_r']
                assert math.isclose(fitted_a*new['omega_rad_s'],new['relaxation_prediction'],rel_tol=1e-14)
                tennis_controls+=1
            for row in fitted['leave_one_out']['records']:
                other=[p for i,p in enumerate(measured['points']) if i!=row['omitted_index']]
                a=sum(p['apparent_omega_rad_s']*p['measured_effective_mu_r'] for p in other)/sum(p['apparent_omega_rad_s']**2 for p in other)
                assert math.isclose(a,row['relaxation_s'],rel_tol=1e-14)
                loo_controls+=1
        assert tennis_controls==loo_controls==17
        rock_source=json.loads((ROOT/'research/restitution-validation-report/fits.json').read_text())['data']
        by_row={r['row']:r for r in rock_source}
        rock_controls=0
        for model in result['rocks']['height_split'].values():
            assert model['training_count']==50 and model['evaluation_count']==25
            for row in model['records']:
                assert by_row[row['source_row']]['release_height_m']==4.5
                assert row['observed']==[by_row[row['source_row']][k] for k in ('vn_after_m_s','vt_after_m_s','omega_after_rad_s')]
                assert 0<=row['effective_e_n']<=1 and row['energy_retained']<=1+1e-12
                rock_controls+=1
        for fold in result['rocks']['angle_folds']:
            for model in fold['models'].values():
                for row in model['records']:
                    assert by_row[row['source_row']]['plate_angle_deg']==fold['angle']
                    assert row['energy_retained']<=1+1e-12
                    rock_controls+=1
        assert rock_controls==200
        rng=np.random.default_rng(701026)
        worst=dict(momentum=0.,energy=0.,semigroup=0.,length_scale=0.,ode=0.)
        for dim in (2,3):
            for _ in range(100):
                mass=10**rng.uniform(-3,2);R=10**rng.uniform(-3,-.3)
                alpha=float(rng.choice([.4,.5,2/3]));omega=rng.uniform(.1,100)
                A=10**rng.uniform(-6,-1);dt=10**rng.uniform(-6,1)
                if dim==2:n,s=np.array([0.,1,0]),np.array([0.,0,1])
                else:q,_=np.linalg.qr(rng.normal(size=(3,3)));n,s=q[:,1],q[:,2]
                # Synthetic static capacity; no measured coefficient invented.
                mu_s=1.1*A*omega/(1+alpha)
                e=np.cross(s,n);I=alpha*mass*R**2
                physical=[]
                for ell in (R*.01,R,R*100):
                    out=rolling_advance(mass,R,alpha,n,s,omega,dt,A,mu_s,ell)
                    v0=R*omega*e;w0=omega*s
                    momentum_error=max(np.linalg.norm(mass*(out['velocity']-v0)-out['linear_impulse'])/(mass*R*omega),
                        np.linalg.norm(I*(out['angular_velocity']-w0)-np.cross(-R*n,out['linear_impulse'])-out['angular_impulse'])/(I*omega))
                    E=.5*mass*np.dot(out['velocity'],out['velocity'])+.5*I*np.dot(out['angular_velocity'],out['angular_velocity'])
                    energy_error=abs(E+out['dissipation']-out['energy_before'])/out['energy_before']
                    assert momentum_error<1e-10 and energy_error<1e-10
                    assert np.linalg.norm(out['velocity']+np.cross(out['angular_velocity'],-R*n))<1e-10*R*omega
                    physical.append(np.r_[out['velocity']/(R*omega),out['angular_velocity']/omega])
                    worst['momentum']=max(worst['momentum'],float(momentum_error));worst['energy']=max(worst['energy'],float(energy_error))
                worst['length_scale']=max(worst['length_scale'],float(np.max(np.abs(np.array(physical)-physical[0]))))
                first=rolling_advance(mass,R,alpha,n,s,omega,dt/2,A,mu_s,R)
                second=rolling_advance(mass,R,alpha,n,s,np.linalg.norm(first['angular_velocity']),dt/2,A,mu_s,R)
                semigroup=float(np.linalg.norm(second['angular_velocity']-out['angular_velocity'])/omega)
                worst['semigroup']=max(worst['semigroup'],semigroup);assert semigroup<1e-10
                # Independent continuous Newton/Euler dynamics eliminated using
                # v=R*omega: (I+m*R^2)*omegadot=-A*N*R*omega.
                if _<10:
                    rate=A*mass*9.81*R/(I+mass*R**2)
                    # Avoid stiff long integration for nearly extinguished cases.
                    checked_dt=min(dt,5/rate)
                    ode=solve_ivp(lambda t,y:[-rate*y[0]],(0,checked_dt),[omega],rtol=2e-11,atol=2e-13)
                    tested=rolling_advance(mass,R,alpha,n,s,omega,checked_dt,A,mu_s,R)
                    err=abs(ode.y[0,-1]-np.linalg.norm(tested['angular_velocity']))/omega
                    worst['ode']=max(worst['ode'],float(err));assert err<1e-9
        unchanged=rolling_advance(1.,.03,.4,[0,1,0],[0,0,1],10.,.1,0.,0.,.001)
        assert np.allclose(unchanged['angular_velocity'],[0,0,10.])
        # Large dt would pass an integrated budget although the initial force is
        # impossible. The model must reject using instantaneous capacity.
        rejected=False
        try:rolling_advance(1.,.03,.4,[0,1,0],[0,0,1],100.,100.,.1,.01,.03)
        except ValueError:rejected=True
        assert rejected
        # Compare one-parameter normal law with the analytically known elastic
        # collision time and weak-damping first-order asymptotic coefficient.
        elastic=normal_reference(0.,rtol=2e-12)
        xmax=(5/4)**.4
        exact_duration=2*xmax/2.5*gamma(.4)*gamma(.5)/gamma(.9)
        assert abs(elastic['restitution']-1)<1e-8
        assert abs(elastic['duration']-exact_duration)<1e-8
        first_order=[]
        for b in (.002,.001,.0005):
            value=normal_reference(b,rtol=2e-12)
            first_order.append((1-value['restitution'])/b)
        assert abs(first_order[-1]-1.15344)<.001
        worst_normal=0.;normal_controls=[]
        # Tight integration and stored-energy accounting at force-zero release.
        for b in np.r_[0,np.geomspace(.001,8,24)]:
            value=normal_reference(float(b),rtol=2e-12)
            normal_controls.append(value)
            worst_normal=max(worst_normal,abs(value['energy_balance_residual']))
            assert abs(value['energy_balance_residual'])<1e-8
            assert value['dissipated_energy']>=0 and value['stored_energy_at_release']>=0
            assert 0<value['restitution']<=1+1e-8 and value['minimum_sampled_force']>-1e-8
            if b>0:assert value['release_compression']>0
        content=np.load(args.evidence/'normal-table.npz');table=NormalTable(content['beta'],content['restitution'])
        for invalid in (-.01,8.01,float('nan')):
            try:table.evaluate(invalid)
            except ValueError:pass
            else:raise AssertionError('table extrapolation not rejected')
        report=dict(pass_=True,rolling_planar_controls=100,rolling_spatial_controls=100,
            coordinate_lengths_per_case=3,rolling_maximum_relative_errors=worst,
            independent_continuous_rolling_controls=20,zero_relaxation_preserves_motion=True,
            insufficient_instantaneous_static_capacity_rejected=True,normal_controls=len(normal_controls),
            normal_maximum_energy_balance_error=worst_normal,
            elastic_collision_time=dict(predicted=elastic['duration'],analytic=exact_duration),
            weak_damping_first_order_ratios=first_order,table_extrapolation_rejected=True,
            original_rolling_records_checked=17,leave_one_rolling_point_out_checked=17,
            rock_records_and_splits_checked=rock_controls,normal_reference_records=normal_controls,
            source_sha256=hashlib.sha256((HERE/'model.py').read_bytes()).hexdigest(),
            full_empirical_collision_validation=False,production_adopted=False,
            source_direction_zero_slip_branch_resolved=False)
        (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        for name in ('audit.py','model.py'):(args.output/name).write_bytes((HERE/name).read_bytes())
        print(json.dumps({k:v for k,v in report.items() if k!='normal_reference_records'},indent=2))
    except Exception as error:
        (args.output/'failure.json').write_text(json.dumps(dict(type=type(error).__name__,reason=str(error)),indent=2)+'\n')
        raise


if __name__=='__main__':main()
