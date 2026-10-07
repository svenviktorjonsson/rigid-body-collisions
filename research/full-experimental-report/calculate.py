"""Fresh, reproducible calculations; preserve all outputs in a new directory."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import shutil
import numpy as np
from rolling import audit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(predicted, observed):
    p, o = np.asarray(predicted, float), np.asarray(observed, float)
    # This is an angle of observable magnitude signatures, not a heading error.
    p, o = abs(p), abs(o)
    if np.linalg.norm(p) == 0 or np.linalg.norm(o) == 0:
        return None
    angle = math.degrees(math.atan2(np.linalg.norm(np.outer(p, o)-np.outer(o, p))/math.sqrt(2), p@o))
    return dict(signature_angle_error_deg=angle,
                signature_size_error_percent=100*(np.linalg.norm(p)/np.linalg.norm(o)-1))


def extract_tennis(pdf):
    import pymupdf
    page = pymupdf.open(pdf)[5]
    # Axis coordinates calibrated against visible tick marks, not fitted lines.
    axes = [
        dict(name='moderately_old', yband=(130, 220), x0=383.861145, x1=490.710250,
             xmin=60., xmax=100., y0=215.620605, y1=133.936646, ymin=.01, ymax=.03,
             published_slope=.00042, published_intercept=-.019),
        dict(name='old', yband=(350, 438), x0=356.131500, x1=518.137146,
             xmin=50., xmax=110., y0=436.877289, y1=352.053650, ymin=.01, ymax=.03,
             published_slope=.00026, published_intercept=-.0023),
        dict(name='new', yband=(580, 678), x0=384.581695, x1=494.422760,
             xmin=60., xmax=100., y0=676.103149, y1=599.658691, ymin=.1, ymax=.4,
             published_slope=.0058, published_intercept=-.28)
    ]
    paths = page.get_drawings()
    datasets = []
    for ax in axes:
        markers = {}
        for path_index, d in enumerate(paths):
            if d['color'] != (0., 0., 1.) or len(d['items']) != 4 or not all(i[0] == 'c' for i in d['items']):
                continue
            rect = d['rect']; x = (rect.x0+rect.x1)/2; y = (rect.y0+rect.y1)/2
            if not ax['yband'][0] <= y <= ax['yband'][1]:
                continue
            key = (round(x, 3), round(y, 3))
            if key in markers:
                markers[key]['duplicate_path_indices'].append(path_index)
                continue
            rpm = ax['xmin']+(x-ax['x0'])*(ax['xmax']-ax['xmin'])/(ax['x1']-ax['x0'])
            mu = ax['ymin']+(ax['y0']-y)*(ax['ymax']-ax['ymin'])/(ax['y0']-ax['y1'])
            markers[key] = dict(apparent_omega_rpm=rpm, apparent_omega_rad_s=rpm*2*math.pi/60,
                                measured_effective_mu_r=mu, pdf_center_pt=[x, y],
                                pdf_path_index=path_index, duplicate_path_indices=[])
        points = sorted(markers.values(), key=lambda p: p['apparent_omega_rpm'])
        assert len(points) == (5 if ax['name'] == 'moderately_old' else 6)
        train = points[::2]; test = points[1::2]
        estimate = max(0., float(np.mean([p['measured_effective_mu_r'] for p in train])))
        for i, p in enumerate(points):
            p.update(split='train' if i % 2 == 0 else 'held_out', constant_mu_r_prediction=estimate,
                     published_curve_prediction=ax['published_slope']*p['apparent_omega_rpm']+ax['published_intercept'])
        rmse = lambda key, group: float(np.sqrt(np.mean([(p[key]-p['measured_effective_mu_r'])**2 for p in group])))
        loo = [float(np.mean([q['measured_effective_mu_r'] for q in points if q is not p])) for p in points]
        datasets.append(dict(specimen=ax['name'], calibration_axis=ax, points=points,
                             estimated_constant_mu_r=estimate, training_count=len(train), holdout_count=len(test),
                             held_out_constant_rmse=rmse('constant_mu_r_prediction', test),
                             held_out_fixed_published_curve_rmse=rmse('published_curve_prediction', test),
                             held_out_zero_moment_rmse=float(np.sqrt(np.mean([p['measured_effective_mu_r']**2 for p in test]))),
                             leave_one_out_estimate_range=[min(loo), max(loo)],
                             coefficient_error_per_pdf_point=(ax['ymax']-ax['ymin'])/(ax['y0']-ax['y1']),
                             valid_apparent_rpm_range=[points[0]['apparent_omega_rpm'], points[-1]['apparent_omega_rpm']],
                             minimum_mu_s_for_no_slip_if_alpha_0_4=estimate/1.4,
                             status='One-parameter effective quasirolling fit; held-out speeds are same-apparatus validation, not independent ball/bounce validation'))
    return dict(source_url='https://arxiv.org/pdf/0809.4823v2', source_sha256=digest(pdf),
                page_1_based=6, dataset_count=3, measured_point_count=17, datasets=datasets,
                source_convention='mu_r=tan(beta), apparent belt-derived ball rpm; source reports slight skidding',
                fit_inputs_include_published_curve=False,
                source_conditions='Three arbitrary non-ITF-approved specimens; reported 32 degrees C and high humidity',
                limitation='Radius, mass, exact rubber formulation, independent sliding/static coefficients and impact restitution not fully specified per specimen. Negative author curve intercepts are apparent-speed/skid behavior, not a negative material friction coefficient; never extrapolate those curves.')


def glass_comparison():
    old = json.loads((ROOT/'research/documented-materials/glass-worksheet-comparison/summary.json').read_text())
    cache = Path('/home/viktor/.cache/physics-documented-materials-20261006/3mmglass-binary-source')
    assert digest(cache) == old['source_sha256']
    profile = next(p for p in json.loads((ROOT/'research/documented-materials/catalog.json').read_text())['profiles'] if p['id']=='glass-soda-binary')
    en, et, mu = [profile[k]['value'] for k in ('normal_restitution', 'tangential_restitution', 'sliding_friction')]
    R = profile['sphere_diameter_m']/2
    mass = 4*math.pi*R**3*profile['sphere_density_kg_m3']/3
    rows = []
    for src in old['records']:
        gn, gt = src['incoming_relative_normal_m_s'], src['incoming_relative_tangent_m_s']
        pn = -(1+en)*gn*mass/2
        pt = float(np.clip(-(1+et)*gt*mass/7, -mu*pn, mu*pn))
        normal, center, contact = -en*gn, gt+2*pt/mass, gt+7*pt/mass
        # Pair energy at zero total linear momentum, both initially unspun.
        spin = 2.5*abs(pt)/(mass*R)
        before = mass*(gn**2+gt**2)/4
        after = mass*(normal**2+center**2)/4+.4*mass*R**2*spin**2
        assert after <= before*(1+1e-12)
        assert max(abs(normal-src['native_normal_after_m_s']), abs(center-src['native_tangent_center_after_m_s']), abs(contact-src['native_tangent_contact_after_m_s'])) < 1e-7
        predicted = [normal, center]
        observed = [src['source_normal_after_m_s'], src['source_tangent_center_after_m_s']]
        rows.append(dict(source_excel_row=src['source_excel_row'], predicted=predicted, observed=observed,
                         predicted_contact_tangent_m_s=contact, reconstructed_contact_tangent_m_s=src['source_tangent_contact_after_m_s'],
                         energy_retained=after/before, energy_change_J=after-before,
                         diagram=metric(predicted, observed), empirical_spin_independent=False))
    err = np.array([np.array(r['predicted'])-r['observed'] for r in rows])
    return dict(case_count=len(rows), fixed_parameters=dict(e_n=en, e_t=et, source_sliding_mu=mu),
                normal_rmse_m_s=float(np.sqrt(np.mean(err[:, 0]**2))),
                center_tangent_rmse_m_s=float(np.sqrt(np.mean(err[:, 1]**2))),
                records=rows, source_sha256=digest(cache), source_url=old['source_url'],
                status='Fresh analytical zero-free-couple comparator, cross-checked against preserved native runs; not general directional rolling/torsion validation',
                limitations=old['limitations'])


def ball_moment_audit():
    inputs = [('golf', 'granite', .87, .03, 14.8), ('golf', 'rubber', .59, .51, 20.5),
              ('golf', 'superball_disk', .81, .55, 23.6), ('golf', 'tennis_strings', .91, -.01, 15.),
              ('superball', 'granite', .78, .49, 14.9), ('superball', 'rubber', .78, .41, 14.5),
              ('superball', 'superball_disk', .78, .57, 18.2), ('superball', 'tennis_strings', .91, -.10, 9.)]
    rows = []
    for ball, surface, en, et, obs in inputs:
        R, mass = (.0214, .045) if ball == 'golf' else (.029, .103)
        alpha, speed, angle = .4, 4., math.radians(25)
        sn, cs = math.sin(angle), math.cos(angle)
        baseline = (1+et)*sn/((1+alpha)*R)
        # omega is positive topspin, i.e. axis -z for x-translation/y-normal.
        required_mu = ((1+et)*sn-(1+alpha)*R*obs)/((1+en)*cs)
        estimated_mu = max(0., required_mu)
        pred = ((1+et)*sn-estimated_mu*(1+en)*cs)/((1+alpha)*R)
        inferred_range = [((1+et+de)*math.sin(math.radians(a))-(1+alpha)*R*(obs+do))/((1+en+dn)*math.cos(math.radians(a)))
                          for a in (24, 26) for de in (-.01, .01) for dn in (-.01, .01) for do in (-.1, .1)]
        pred_state = np.array([en*speed*cs, speed*(R*pred-et*sn), R*speed*pred])
        obs_state = np.array([en*speed*cs, speed*(R*obs-et*sn), R*speed*obs])
        pn = mass*(1+en)*speed*cs
        pt = mass*(pred_state[1]-speed*sn)
        couple = -estimated_mu*R*pn
        omega = np.array([0., 0, -speed*pred])
        vminus = np.array([speed*sn, -speed*cs, 0.])
        vplus = np.array([pred_state[1], pred_state[0], 0.])
        r = np.array([0., -R, 0.]); p = np.array([pt, pn, 0.]); s = np.array([0., 0., -1.])
        t = vminus/np.linalg.norm(vminus)
        # Preserve t = full relative contact velocity direction. The resulting
        # n,t components are not equal to geometric normal/tangent impulses.
        scalar_pt = pt/t[0]; scalar_pn = pn-scalar_pt*t[1]
        assert np.allclose(p, scalar_pn*np.array([0,1,0])+scalar_pt*t)
        I = alpha*mass*R**2
        assert np.allclose(I*omega, np.cross(r, p)+couple*s)
        assert abs(vplus[0]-R*speed*pred+et*speed*sn) < 1e-12
        before = .5*mass*speed**2
        after = .5*mass*(vplus@vplus)+.5*I*(omega@omega)
        work = p@vminus+.5*(p@p/mass + (np.cross(r,p)+couple*s)@(np.cross(r,p)+couple*s)/I)
        assert abs(after-before-work) < 1e-12
        assert after <= before*(1+1e-12)
        rows.append(dict(ball=ball, surface=surface, radius_m=R, mass_kg=mass, assumed_alpha=alpha,
                         fixed_e_n=en, fixed_e_t=et, observed_spin_factor_rad_m=obs,
                         force_only_spin_factor_rad_m=baseline, estimated_mu_r=estimated_mu,
                         required_unconstrained_mu_r=required_mu, required_mu_r_error_envelope=[min(inferred_range), max(inferred_range)],
                         full_moment_spin_factor_rad_m=pred, residual_rad_m=pred-obs,
                         passive_rolling_explanation_possible_at_nominal=required_mu >= 0,
                         passive_rolling_explanation_possible_with_error_limits=max(inferred_range) >= 0,
                         energy_retained=after/before, energy_change_J=after-before,
                         independent_angular_impulse_N_m_s=couple,
                         required_net_friction_impulse_ratio=abs(pt)/pn,
                         direction_components=dict(delta_p_n=scalar_pn, delta_p_t=scalar_pt, delta_L_s=couple, delta_L_n=0.),
                         predicted_observable_signature=pred_state.tolist(), conditional_measured_signature=obs_state.tolist(),
                         diagram=metric(pred_state, obs_state),
                         status='Same-outcome one-parameter angular-impulse calibration under assumed inertia; negative mu_r rejected, no independent validation',
                         missing=['measured central inertia', 'independent same-pair mu_s and mu_d', 'force/moment history', 'general mixed angular closure']))
    # Separate normal-pressure moment hypothesis: a signed physical offset can
    # assist spin through redistribution of contact work without negative friction.
    # Try one shared dimensionless offset per ball on THREE surfaces, then test on
    # the fourth. This cross-surface assumption is itself tested, not accepted.
    for row in rows:
        cohort = [r for r in rows if r['ball']==row['ball'] and r is not row]
        A = np.array([(1+r['fixed_e_n'])*math.cos(math.radians(25))/(1.4*r['radius_m']) for r in cohort])
        discrepancy = np.array([r['force_only_spin_factor_rad_m']-r['observed_spin_factor_rad_m'] for r in cohort])
        offset = float(np.clip(A@discrepancy/(A@A), -1., 1.))
        sensitivity = (1+row['fixed_e_n'])*math.cos(math.radians(25))/(1.4*row['radius_m'])
        pred = row['force_only_spin_factor_rad_m']-sensitivity*offset
        vx = row['radius_m']*pred-row['fixed_e_t']*math.sin(math.radians(25))
        energy = (row['fixed_e_n']*math.cos(math.radians(25)))**2+vx**2+.4*(row['radius_m']*pred)**2
        row['held_out_signed_moment'] = dict(training_surfaces=[r['surface'] for r in cohort],
            estimated_shared_offset_over_R=offset, physical_offset_m=offset*row['radius_m'],
            predicted_spin_factor_rad_m=pred, residual_rad_m=pred-row['observed_spin_factor_rad_m'],
            energy_retained=energy, gross_sphere_radius_bound_pass=abs(offset)<=1,
            actual_contact_patch_bound_known=False, total_passivity_pass=energy<=1+1e-12,
            status='One shared signed pressure-moment offset per ball fitted on three surfaces; fourth spin not used in fit. Counterface transfer is an unqualified hypothesis.')
        calibrated_energy=(row['fixed_e_n']*math.cos(math.radians(25)))**2+(row['radius_m']*row['observed_spin_factor_rad_m']-row['fixed_e_t']*math.sin(math.radians(25)))**2+.4*(row['radius_m']*row['observed_spin_factor_rad_m'])**2
        row['singleton_signed_offset'] = dict(offset_over_R=row['required_unconstrained_mu_r'],
            physical_offset_m=row['required_unconstrained_mu_r']*row['radius_m'],
            energy_retained=calibrated_energy, gross_sphere_radius_bound_pass=abs(row['required_unconstrained_mu_r'])<=1,
            status='Inverse mechanics from same measured spin; exact agreement is calibration, not prediction. Signed offset is not mu_r.')
    return dict(source_url='https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/48.%20EnhanceBounce.pdf',
                table='I, all eight oblique rows; two vertical COR-only rows are not independent outcome tests',
                records=rows, case_count=8, all_energy_pass=True,
                leave_one_surface_out=dict(force_only_spin_rmse_rad_m=float(np.sqrt(np.mean([(r['force_only_spin_factor_rad_m']-r['observed_spin_factor_rad_m'])**2 for r in rows]))),
                    signed_moment_spin_rmse_rad_m=float(np.sqrt(np.mean([r['held_out_signed_moment']['residual_rad_m']**2 for r in rows]))),
                    worst_force_only_error_rad_m=max(abs(r['force_only_spin_factor_rad_m']-r['observed_spin_factor_rad_m']) for r in rows),
                    worst_signed_moment_error_rad_m=max(abs(r['held_out_signed_moment']['residual_rad_m']) for r in rows),
                    parameter_count_per_ball_per_fold=1,
                    all_passive=all(r['held_out_signed_moment']['total_passivity_pass'] for r in rows)),
                explanation='Fixed measured e_n/e_t and angle; approximate 4 m/s only sets displayed dimensional states. Spin factor is speed-independent. Scalar impulses use full incoming t; zero incoming spin uses the planar force-induced spin onset axis. These are reduced instantaneous endpoint calculations, not native trajectories. Tangential output is reconstructed from author e_t and measured S, not a separate raw-velocity validation.',
                fit_interpretation='A fitted negative resistance would be an assisting torque and is rejected here. A real deformable patch can return elastic energy through a moment, so this rejects a purely dissipative rolling explanation, not all independent angular impulses.')


def rock_comparison():
    old_path = ROOT/'research/restitution-validation-report/fits.json'
    old = json.loads(old_path.read_text())
    source = Path('/home/viktor/.cache/physics-public-rock-data-20261006/wang2018/extracted/data set of nhess-2018-108.xlsx')
    assert digest(source) == old['plan']['input_xlsx_sha256']
    en, et, mu = old['models']['fixed']['fit']['theta']
    rows = []
    for r in old['data']:
        R = r['diameter_m']/2; vn, vt = r['vn_before_m_s'], r['vt_before_m_s']
        p_t_per_mass = float(np.clip(-(1+et)*vt/3.5, -mu*(1+en)*vn, mu*(1+en)*vn))
        predicted = [en*vn, vt+p_t_per_mass, 2.5*abs(p_t_per_mass)/R]
        obs = [r['vn_after_m_s'], r['vt_after_m_s'], r['omega_after_rad_s']]
        # CONDITIONAL inverse momentum under same sphere/coplanar/topspin assumptions.
        required_mu_r = -(.4*R*obs[2]+obs[1]-vt)/(vn+obs[0])
        before_per_mass = .5*(vn**2+vt**2)
        after_per_mass = .5*(predicted[0]**2+predicted[1]**2)+.2*R**2*predicted[2]**2
        assert after_per_mass <= before_per_mass*(1+1e-12)
        rows.append(dict(source_row=r['row'], split='held_out' if r['release_height_m']==4.5 else 'train',
                         diameter_m=r['diameter_m'], plate_angle_deg=r['plate_angle_deg'], release_height_m=r['release_height_m'],
                         predicted_historical_sphere_proxy=predicted, measured_magnitudes=obs,
                         conditional_required_mu_r=required_mu_r, proxy_energy_retained=after_per_mass/before_per_mass))
    test = [r for r in rows if r['split']=='held_out']
    err = np.array([np.array(r['predicted_historical_sphere_proxy'])-r['measured_magnitudes'] for r in test])
    rms = np.sqrt(np.mean(err**2, axis=0))
    assert np.allclose(rms, [old['models']['fixed']['heldout'][k] for k in ('normal_velocity_rmse_m_s','tangent_velocity_rmse_m_s','angular_speed_rmse_rad_s')])
    return dict(case_count=75, training_count=50, held_out_count=25, source_url=old['plan']['experimental_source'],
                source_sha256=digest(source), fixed_historical_estimates=dict(e_n=en, e_t=et, sliding_mu=mu),
                held_out_rmse=dict(normal_m_s=float(rms[0]), tangent_m_s=float(rms[1]), angular_rad_s=float(rms[2])),
                records=rows, inference_is_true_rolling_friction=False,
                status='Recomputed historical zero-free-couple sphere proxy; not a prediction of actual rock geometry or requested full-angular model',
                limitations=old['plan']['assumptions'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--tennis-pdf', type=Path, default=Path('/home/viktor/.cache/physics-rolling-source-audit-20261006/tennis2008.pdf'))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    result = dict(plan=json.loads((HERE/'plan.json').read_text()), rolling_mechanics=audit(),
                  tennis=extract_tennis(args.tennis_pdf), glass=glass_comparison(), balls=ball_moment_audit(), rocks=rock_comparison())
    catalog = json.loads((ROOT/'research/documented-materials/catalog.json').read_text())
    result['material_readiness'] = [dict(profile_id=p['id'], e_n=p['normal_restitution']['value'], e_t=p['tangential_restitution']['value'],
                                         documented_sliding_mu=p['sliding_friction']['value'], independently_documented_mu_s=None,
                                         independently_documented_mu_r=None, source_url=p['source_url'],
                                         full_moment_profile_qualified=False) for p in catalog['profiles']]
    result['scope'] = dict(collision_records=107, rolling_measurements=17,
                           unique_physical_events_confirmed=False, # summary coefficients/worksheets can share characterization trials
                           full_directional_native_engine_validated=False,
                           independent_measured_3d_shape_and_signed_spin_comparisons=0,
                           all_13_rapid_irregular_case_baseline_pass=False,
                           main_limit='General angular/friction branch closure and matched physical parameters remain incomplete; no claim that all numbers match.')
    tracked = [HERE/'plan.json', HERE/'calculate.py', HERE/'rolling.py', ROOT/'research/documented-materials/catalog.json',
               ROOT/'research/documented-materials/glass-worksheet-comparison/summary.json', ROOT/'research/restitution-validation-report/fits.json']
    result['input_hashes'] = {str(p.relative_to(ROOT)): digest(p) for p in tracked}
    snapshot=args.output/'source-snapshot'; snapshot.mkdir()
    for p in (HERE/'plan.json', HERE/'calculate.py', HERE/'rolling.py'):
        shutil.copyfile(p, snapshot/p.name)
    (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    # Downloadable row-level comparison tables, with calibration status intact.
    for name in ('glass', 'balls', 'rocks'):
        records = result[name]['records']; keys = list(records[0])
        with (args.output/f'{name}.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, keys); w.writeheader()
            w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in records])
    print(json.dumps(dict(rolling=result['rolling_mechanics'], glass_rmse=[result['glass']['normal_rmse_m_s'],result['glass']['center_tangent_rmse_m_s']],
                          tennis=[{k:d[k] for k in ('specimen','estimated_constant_mu_r','held_out_constant_rmse','held_out_fixed_published_curve_rmse')} for d in result['tennis']['datasets']],
                          balls=[{k:r[k] for k in ('ball','surface','required_unconstrained_mu_r','residual_rad_m')} for r in result['balls']['records']],
                          rock_rmse=result['rocks']['held_out_rmse']), indent=2))


if __name__ == '__main__':
    main()
