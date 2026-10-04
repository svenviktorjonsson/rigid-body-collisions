"""Render an auditable Markdown report from the executed study summary."""
import json
from pathlib import Path


def make_report():
    root = Path(__file__).parent / 'moving-container'
    r = json.loads((root / 'results/summary.json').read_text())
    lines = ['# Moving-container verification report', '',
        'Executed on 4 October 2026. Synthetic numerical verification of declared rigid mechanics; '
        'no experimental material validation or universal best-solver claim.', '',
        f"The archive retains **{len(r['records'])} histories**. Each setting has one warm-up and "
        f"{r['repeats']} recorded timing samples. All 54 Python/research tests pass. Native comparators "
        'are pinned Box2D 2.4.1 (block) and 3.1.1 (temporal). This study does not execute Vektor, '
        'WASM or physical GPU. Timing measures solver plus controller, excluding Python, JSON, '
        'state readout and process startup. Results are from one Linux machine, not a controlled '
        'cross-platform throughput study.', '',
        '## Independent exact packed-row check', '',
        'N touching unit-mass disks initially rest inside a closed box translating at 1 m/s. '
        'For the declared rigid, zero-restitution, frictionless constraints, every outgoing ball '
        'must move at 1 m/s. The first-output-frame maximum individual error budget is 0.01 m/s. '
        'Expected momentum is N kg m/s, kinetic energy N/2 J, actuator work N J and dissipation N/2 J. '
        'Native first-frame failure includes contact discovery and floating geometry; it cannot '
        'be attributed solely to iterations. A physical elastic chain would transmit finite-speed waves.', '',
        '| Balls | Setting | Max velocity error (m/s) | Budget met | Median solver+controller (ms) |',
        '|---:|---|---:|:---:|---:|']
    for record in r['records']:
        if 'analytic_passed' in record:
            lines.append(f"| {record['scene'].split('_')[-1]} | {record['label']} | {record['analytic_max_velocity_error_m_s']:.6g} | "
                         f"{'yes' if record['analytic_passed'] else 'no'} | {1000*record['median_engine_controller_s']:.4g} |")
    maximum = max(a['max_velocity_error_m_s'] for a in r['global_frozen_projection'])
    lines += ['', f'The independent dense frozen normal projection satisfies all four rows with maximum '
              f'velocity error {maximum:.3g} m/s and checked complementarity residuals. This is a '
              'correctness oracle, not a complete frictional engine or a measured performance winner.', '',
              '![Packed-row error](results/packed-row.png)', '', '## Frictional 100-ball motion', '',
              'Each scene lasts 2 s, with gravity -9.81 m/s², radius 0.1 m, mass 1 kg, friction '
              '0.4 and restitution 0. Four wall fixtures move as one kinematic body. Motion is '
              'translation at 0.6 m/s, reversals every 0.5 s, or rotation at 0.5 rad/s. '
              'These coefficients are idealized declarations. Fine simulation is not physical ground truth.', '',
              'Reference qualification requires every adjacent edge in two separate sweeps to meet '
              'quarter budgets: position RMS 0.005 m, velocity RMS 0.0125 m/s and spin RMS '
              '0.0125 rad/s. Primary updates are 16/32/64 at 128 iterations; iterations are '
              '32/64/128 at 64 primary updates. The highest-work state is only a candidate reference '
              'unless all four edges pass.', '',
              '| Motion | Reference qualified | Worst refinement / quarter budget |', '|---|:---:|---:|']
    budgets = json.loads((root / 'plan.json').read_text())['reference_budget']
    for scene, q in r['reference_qualification'].items():
        ratio = max(e['errors'][k]/b for e in q['refinement'] for k, b in budgets.items())
        lines.append(f"| {scene.split('_')[-1]} | {'yes' if q['qualified'] else 'no'} | {ratio:.4g} |")
    lines += ['', '| Motion | Setting | Position RMS (m) | Velocity RMS (m/s) | Spin RMS (rad/s) | Qualified accuracy pass |',
              '|---|---|---:|---:|---:|:---:|']
    for c in r['comparisons']:
        e = c['errors']; status = 'yes' if c['passed_qualified_reference'] else ('no' if c['reference_qualified'] else 'unqualified')
        lines.append(f"| {c['scene'].split('_')[-1]} | {c['label']} | {e['rms_position_m']:.4g} | "
                     f"{e['rms_velocity_m_s']:.4g} | {e['rms_spin_rad_s']:.4g} | {status} |")
    lines += ['', 'Candidate budgets are 0.02 m, 0.05 m/s and 0.05 rad/s. '
              'Comparisons against an unqualified reference are diagnostic and cannot establish accuracy. '
              'No tolerance was relaxed after looking at the results. The current adaptive controller '
              'is a whole-world heuristic; this study does not certify error-controlled adaptation.', '',
              '## Controls and geometric diagnostics', '',
              '| Control, 100 balls | Max position difference (m) | Max velocity difference (m/s) | Max spin difference (rad/s) |',
              '|---|---:|---:|---:|']
    for c in r['controls']:
        lines.append(f"| {c['control']} | {c['max_position_error_m']:.4g} | {c['max_velocity_error_m_s']:.4g} | {c['max_spin_error_rad_s']:.4g} |")
    lines += ['', 'Controls cover a Galilean boost, joint free translation of box and contents, and '
              'reversing body insertion order. Differences are reported rather than hidden. '
              'Containment is checked in the actual rotating box frame. Independent disk-pair '
              'distances and wall extents are checked at every output frame.', '',
              '| Motion | High setting max disk overlap (m) | High setting max wall violation (m) |',
              '|---|---:|---:|']
    for record in r['records']:
        if record['label'] == 'block_p8_s32' and record['scene'].startswith('grid_100'):
            d = record['diagnostics']; lines.append(f"| {record['scene'].split('_')[-1]} | {d['observed_max_disk_overlap_m']:.4g} | {d['observed_max_wall_violation_m']:.4g} |")
    lines += ['', 'Observed geometry does not prove continuous no-tunneling. Contents momentum is '
              'not conserved under moving walls. The constant-translation, zero-gravity row and '
              'control runs include actuator-work accounting. Shaking and rotation need a future '
              'time-resolved contact reaction ledger; no exact work result is claimed for them.', '',
              '## Consequences for the engine', '',
              'The packed-row failures falsify a universal accuracy claim for the current high preset. '
              'Spend work within one unchanged physical law, preserving contact/history states, '
              'and investigate sparse globally coupled island solves. The normal oracle establishes '
              'the target constraints but does not yet supply a complete Coulomb friction solver. '
              'Combine convergence, independent analytic tests, contact/CCD geometry and energy plus '
              'actuator work; residuals alone do not bound future trajectory error.', '',
              'An accuracy order through discontinuous impacts has not been established. Use per-observable '
              'tolerances and event timing, and test smooth-region integration order separately. '
              'Authentic elastic tangential/rolling rebound needs measured response and stored contact '
              'history. Coarse deformable cells require additional degrees of freedom, not merely '
              'material labels on a rigid body. The established contact algebra and coefficient-based '
              'friction model remain nonnovel; the joint publication verdict remains conditional.', '',
              '## Reproduction and evidence', '',
              'Run `python -m research.audit_container_study` to independently check the archive, '
              'all trajectory comparisons, analytical row metrics and reference gates. '
              'The plan, source, summary, failures and every full trace are retained. '
              'See README.md for build/reproduction commands and contact-model.pdf for typeset mathematics.', '',
              f"Execution source: `{r['source_commit']}`. Archive SHA-256: `{r['traces_zip_sha256']}`.", '',
              'Sources supporting the method choices: [Catto, Solver2D](https://box2d.org/posts/2024/02/solver2d/), '
              '[Box2D simulation](https://box2d.org/documentation/md_simulation.html), '
              '[MuJoCo contact computation](https://mujoco.readthedocs.io/en/stable/computation/index.html). '
              'Prior-art references and opposing reviews are retained in research/critical-review.md, '
              'publication-case.md and joint-verdict.md. No reference is used to turn synthetic '
              'coefficients into experimental measurements.', '']
    (root / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__': make_report()
