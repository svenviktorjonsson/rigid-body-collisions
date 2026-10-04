"""Reproducible moving-container verification. No material calibration claim.

python -m research.run_container_study --repeats 3
An independent plan is committed before executing the study.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from rigid_engine import BOX2D_COMMITS, run
from research.container_scenes import grid, row, packed_row_contact_system
from research.contact_solver import inelastic_normal_solve
from research.run_rigid_study import errors, normalized_error

ROOT = Path(__file__).parent / 'moving-container'


def diagnostics(scene, result):
    """Independent geometry at observed frames, including rotating container."""
    state = np.asarray(result['states']); box = np.asarray(result['kinematic_states'])[:, 0]
    radius = scene['container']['radius_m']; half = np.asarray(scene['container']['half_extents_m'])
    wall_violation = 0.; overlap = 0.
    for particles, boundary in zip(state, box):
        p = particles[:, :2] - boundary[:2]
        c, s = np.cos(boundary[2]), np.sin(boundary[2])
        local = p @ np.array([[c, -s], [s, c]])
        wall_violation = max(wall_violation, float(np.max(np.abs(local) + radius - half)))
        pairs = cKDTree(p).query_pairs(2 * radius, output_type='ndarray')
        if len(pairs):
            overlap = max(overlap, float(np.max(2 * radius - np.linalg.norm(p[pairs[:, 0]] - p[pairs[:, 1]], axis=1))))
    m, inertia = np.asarray(result['mass']), np.asarray(result['inertia'])
    kinetic = .5 * np.sum(m[None, :] * np.sum(state[:, :, 3:5]**2, axis=2) + inertia[None, :] * state[:, :, 5]**2, axis=1)
    output = {'observed_max_wall_violation_m': wall_violation,
              'observed_max_disk_overlap_m': overlap,
              'kinetic_final_J': float(kinetic[-1]),
              'geometry_scope': 'Observed frames, not a continuous-time no-tunneling proof'}
    # Exact external-work identity only for constant pure translation and zero gravity.
    if not any(scene['gravity']) and not scene['bodies'][0].get('velocity_schedule') and not scene['bodies'][0].get('omega', 0):
        momentum = np.sum(m[None, :, None] * state[:, :, 3:5], axis=1)
        work = (momentum - momentum[0]) @ np.asarray(scene['bodies'][0].get('velocity', [0, 0]))
        dissipation = work - (kinetic - kinetic[0])
        output.update({'actuator_work_final_J': float(work[-1]),
                       'inferred_dissipation_final_J': float(dissipation[-1]),
                       'minimum_inferred_dissipation_J': float(np.min(dissipation))})
    return output


def study(repeats):
    plan = json.loads((ROOT / 'plan.json').read_text())
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    traces = {}; records = []

    def execute(scene, label, backend='block', primary=8, solver=32, policy=None):
        key = scene['id'] + '__' + label
        # One warm-up, then repeated calls. Median covers solver+controller only.
        run(scene, backend=backend, primary_steps=primary, substeps=solver, policy=policy)
        samples = [run(scene, backend=backend, primary_steps=primary, substeps=solver, policy=policy) for _ in range(repeats)]
        result = samples[-1]; traces[key] = result
        record = {'trace': key, 'scene': scene['id'], 'label': label, 'backend': backend,
                  'primary': primary, 'solver': solver, 'policy': policy,
                  'engine_controller_samples_s': [r['engine_and_controller_s'] for r in samples],
                  'median_engine_controller_s': statistics.median(r['engine_and_controller_s'] for r in samples),
                  'diagnostics': diagnostics(scene, result)}
        records.append(record)
        print(key, record['median_engine_controller_s'], flush=True)
        return result, record

    analytic = []
    for n in plan['packed_counts']:
        scene = row(n)
        for backend, p, s in plan['packed_modes']:
            label = f'{backend}_p{p}_s{s}'
            r, record = execute(scene, label, backend, p, s)
            post = np.asarray(r['states'])[-1, :, 3:5]
            error = float(np.max(np.linalg.norm(post - [1, 0], axis=1)))
            record['analytic_max_velocity_error_m_s'] = error
            record['analytic_passed'] = error <= plan['packed_velocity_budget_m_s']
        inverse, G, velocity = packed_row_contact_system(n)
        post, impulses, residual = inelastic_normal_solve(inverse, G, velocity)
        analytic.append({'count': n, 'max_velocity_error_m_s': float(np.max(np.abs(post[:-3:3] - 1))),
                         'residual': residual, 'wall_impulse_kg_m_s': float(impulses[0] - impulses[-3])})

    comparisons = []; qualified = {}
    for motion in plan['grid_motions']:
        scene = grid(plan['grid_side'], motion, duration=plan['grid_duration_s'])
        runs = {}
        for p, s in plan['reference_modes']:
            runs[p, s], _ = execute(scene, f'reference_p{p}_s{s}', primary=p, solver=s)
        reference = runs[64, 128]
        edges = [((16, 128), (32, 128)), ((32, 128), (64, 128)),
                 ((64, 32), (64, 64)), ((64, 64), (64, 128))]
        refinement = [{'from': list(a), 'to': list(b), 'errors': errors(runs[b], runs[a])} for a, b in edges]
        q = all(normalized_error(edge['errors'], plan['reference_budget']) <= 1 for edge in refinement)
        qualified[scene['id']] = {'qualified': q, 'refinement': refinement}
        for backend, p, s in plan['grid_modes']:
            label = f'{backend}_p{p}_s{s}'
            candidate, record = execute(scene, label, backend, p, s)
            error = errors(reference, candidate)
            comparisons.append({'scene': scene['id'], 'label': label, 'reference_qualified': q,
                                'errors': error, 'within_budget_against_candidate_reference': normalized_error(error, plan['budget']) <= 1,
                                'passed_qualified_reference': q and normalized_error(error, plan['budget']) <= 1})
        candidate, record = execute(scene, 'block_adaptive', policy={})
        error = errors(reference, candidate)
        comparisons.append({'scene': scene['id'], 'label': 'block_adaptive', 'reference_qualified': q,
                            'errors': error, 'within_budget_against_candidate_reference': normalized_error(error, plan['budget']) <= 1,
                            'passed_qualified_reference': q and normalized_error(error, plan['budget']) <= 1})

    # Independent invariance checks. Counterpart scenes share physical law but not setup hash.
    controls = []
    base = grid(10, 'translate', duration=.25, gravity=(0, 0), friction=0)
    boosted = grid(10, 'translate', duration=.25, boost=(.4, -.2), gravity=(0, 0), friction=0)
    boosted['id'] += '_boosted'
    a, _ = execute(base, 'frame_control'); b, _ = execute(boosted, 'frame_control')
    sa, sb = np.asarray(a['states']), np.asarray(b['states']); times = np.asarray(a['times'])
    controls.append({'control': 'galilean_100',
        'max_position_error_m': float(np.max(np.linalg.norm(sb[:, :, :2] - [.4, -.2] * times[:, None, None] - sa[:, :, :2], axis=2))),
        'max_velocity_error_m_s': float(np.max(np.linalg.norm(sb[:, :, 3:5] - [.4, -.2] - sa[:, :, 3:5], axis=2))),
        'max_spin_error_rad_s': float(np.max(np.abs(sb[:, :, 5] - sa[:, :, 5])))})
    co = grid(10, 'stationary', duration=.25, boost=(.6, -.2), gravity=(0, 0), friction=0)
    co['id'] += '_comoving'; r, _ = execute(co, 'free_comoving')
    s = np.asarray(r['states']); t = np.asarray(r['times'])
    controls.append({'control': 'comoving_100', 'max_position_error_m': float(np.max(np.abs(s[:, :, :2] - s[0, :, :2] - t[:, None, None] * [.6, -.2]))),
                     'max_velocity_error_m_s': float(np.max(np.abs(s[:, :, 3:5] - [.6, -.2]))),
                     'max_spin_error_rad_s': float(np.max(np.abs(s[:, :, 5])))})
    permuted = copy.deepcopy(base); permuted['id'] += '_reverse_body_order'; permuted['bodies'][1:] = permuted['bodies'][1:][::-1]
    r, _ = execute(permuted, 'order_control'); s = np.asarray(r['states'])[:, ::-1]
    controls.append({'control': 'reverse_body_order_100',
                     'max_position_error_m': float(np.max(np.linalg.norm(s[:, :, :2] - sa[:, :, :2], axis=2))),
                     'max_velocity_error_m_s': float(np.max(np.linalg.norm(s[:, :, 3:5] - sa[:, :, 3:5], axis=2))),
                     'max_spin_error_rad_s': float(np.max(np.abs(s[:, :, 5] - sa[:, :, 5])))})

    archive = out / 'traces.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, trace in traces.items():
            z.writestr(name + '.json', json.dumps(trace, separators=(',', ':')))
    report = {'schema_version': 1, 'evidence': 'Synthetic numerical verification, not material validation',
              'plan_sha256': hashlib.sha256((ROOT / 'plan.json').read_bytes()).hexdigest(),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'machine': platform.platform(), 'backend_commits': BOX2D_COMMITS, 'repeats': repeats,
              'records': records, 'global_frozen_projection': analytic, 'reference_qualification': qualified,
              'comparisons': comparisons, 'controls': controls,
              'traces_zip_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    fig, ax = plt.subplots(figsize=(8, 4))
    for backend, p, s in plan['packed_modes']:
        label = f'{backend}_p{p}_s{s}'
        rows = [r for r in records if r['label'] == label and r['scene'].startswith('packed_row')]
        ax.plot(plan['packed_counts'], [r['analytic_max_velocity_error_m_s'] for r in rows], 'o-', label=label)
    ax.axhline(plan['packed_velocity_budget_m_s'], color='black', ls='--', label='0.01 m/s budget')
    ax.set(xlabel='Balls in a closed driven row', ylabel='Maximum individual velocity error [m/s]',
           title='First output frame: exact rigid constraint requires every ball at 1 m/s')
    ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / 'packed-row.png', dpi=160)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1: parser.error('Positive repeats required')
    study(args.repeats)
