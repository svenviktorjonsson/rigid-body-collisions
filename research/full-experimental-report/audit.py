"""Independent direct momentum/energy and data/provenance checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('evidence',type=Path); args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    data=json.loads((args.evidence/'results.json').read_text())
    for rel,expected in data['input_hashes'].items():
        actual=hashlib.sha256((root/rel).read_bytes()).hexdigest()
        assert actual==expected, rel
    for name in ('plan.json','calculate.py','rolling.py'):
        assert (args.evidence/'source-snapshot'/name).read_bytes()==(Path(__file__).parent/name).read_bytes()
    count=0
    for r in data['balls']['records']:
        R,m=r['radius_m'],r['mass_kg']; en,et=r['fixed_e_n'],r['fixed_e_t']
        normal=np.array([0.,1,0]); rvec=-R*normal; inertia=.4*m*R**2
        velocity=np.array([4*math.sin(math.radians(25)),-4*math.cos(math.radians(25)),0.])
        j=np.array([m*(r['predicted_observable_signature'][1]-velocity[0]),m*(r['predicted_observable_signature'][0]-velocity[1]),0.])
        couple=np.array([0.,0.,-r['independent_angular_impulse_N_m_s']])
        out_v=velocity+j/m; out_w=(np.cross(rvec,j)+couple)/inertia
        assert abs(np.linalg.norm(out_w)/4-r['full_moment_spin_factor_rad_m'])<1e-10
        assert abs(out_v[0]+R*out_w[2]+et*velocity[0])<1e-10
        E0=.5*m*(velocity@velocity); E1=.5*m*(out_v@out_v)+.5*inertia*(out_w@out_w)
        assert abs(E1/E0-r['energy_retained'])<1e-12
        assert E1<=E0*(1+1e-12)
        # Nonorthogonal directions retain t=full incoming velocity.
        t=velocity/np.linalg.norm(velocity); scalars=r['direction_components']
        assert np.allclose(j,scalars['delta_p_n']*normal+scalars['delta_p_t']*t)
        # Holdout fit calculated using exactly the other three surface outcomes.
        cohort=[q for q in data['balls']['records'] if q['ball']==r['ball'] and q['surface']!=r['surface']]
        assert set(r['held_out_signed_moment']['training_surfaces'])=={q['surface'] for q in cohort}
        terms=[((1+q['fixed_e_n'])*math.cos(math.radians(25))/(1.4*R),q['force_only_spin_factor_rad_m']-q['observed_spin_factor_rad_m']) for q in cohort]
        offset=sum(a*b for a,b in terms)/sum(a*a for a,b in terms)
        assert abs(offset-r['held_out_signed_moment']['estimated_shared_offset_over_R'])<1e-12
        count+=1
    tennis_count=0
    for d in data['tennis']['datasets']:
        points=d['points']; train=[x for x in points if x['split']=='train']; held=[x for x in points if x['split']=='held_out']
        assert not {x['pdf_path_index'] for x in train}&{x['pdf_path_index'] for x in held}
        assert abs(np.mean([x['measured_effective_mu_r'] for x in train])-d['estimated_constant_mu_r'])<1e-12
        assert all(x['measured_effective_mu_r']>0 for x in points)
        assert abs(np.sqrt(np.mean([(x['constant_mu_r_prediction']-x['measured_effective_mu_r'])**2 for x in held]))-d['held_out_constant_rmse'])<1e-12
        assert d['held_out_fixed_published_curve_rmse']<d['held_out_constant_rmse']
        tennis_count+=len(points)
    assert tennis_count==17 and len(data['glass']['records'])==24 and len(data['rocks']['records'])==75
    assert len([r for r in data['rocks']['records'] if r['split']=='held_out'])==25
    assert all(not p['full_moment_profile_qualified'] for p in data['material_readiness'])
    report=dict(pass_=True,source_hashes_pass=True,source_snapshots_pass=True,
                ball_direct_momentum_and_energy_pass_count=count,nonorthogonal_t_reconstruction_pass=True,
                leave_one_surface_out_direct_spin_fit_separation_pass=True,
                ball_inputs_include_outcome_derived_restitution=True,
                fixed_author_tennis_curves_were_fitted_to_same_measurements=True,
                tennis_data_and_split_pass_count=tennis_count,
                collision_records=107,rolling_points=17,rolling_mechanics=data['rolling_mechanics'],
                scientific_scope_labels_pass=True,full_engine_empirical_validation_claimed=False)
    (args.evidence/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
