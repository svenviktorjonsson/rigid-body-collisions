"""Audit retained traces, published metrics and reference gates independently."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np


def audit():
    root = Path(__file__).parent / 'moving-container'
    plan = json.loads((root / 'plan.json').read_text())
    report = json.loads((root / 'results/summary.json').read_text())
    archive = root / 'results/traces.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == report['traces_zip_sha256']
    assert hashlib.sha256((root / 'plan.json').read_bytes()).hexdigest() == report['plan_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        traces = {Path(name).stem: json.loads(z.read(name)) for name in z.namelist()}
    assert len(traces) == len(report['records']) == 53
    for record in report['records']:
        trace = traces[record['trace']]; s = np.asarray(trace['states'])
        assert np.isfinite(s).all() and s.shape[2] == 6
        assert np.asarray(trace['kinematic_states']).shape == (len(s), 1, 6)
        assert np.all(np.asarray(trace['mass']) > 0) and np.all(np.asarray(trace['inertia']) > 0)
        assert record['backend'] == trace['numerical_model']['backend']
        assert np.isclose(record['median_engine_controller_s'], np.median(record['engine_controller_samples_s']))
        if 'analytic_passed' in record:
            error = np.max(np.linalg.norm(s[-1, :, 3:5] - [1, 0], axis=1))
            assert np.isclose(error, record['analytic_max_velocity_error_m_s'])
            assert record['analytic_passed'] == (error <= plan['packed_velocity_budget_m_s'])
            m = np.asarray(trace['mass']); inertia = np.asarray(trace['inertia'])
            kinetic = .5 * np.sum(m * np.sum(s[-1, :, 3:5]**2, axis=1) + inertia * s[-1, :, 5]**2)
            work = np.sum(m * (s[-1, :, 3] - s[0, :, 3]))
            assert np.isclose(work - kinetic, record['diagnostics']['inferred_dissipation_final_J'], atol=1e-6)
    for comparison in report['comparisons']:
        scene = comparison['scene']; a = np.asarray(traces[scene + '__reference_p64_s128']['states'])
        b = np.asarray(traces[scene + '__' + comparison['label']]['states']); d = b - a
        error = {'rms_position_m': np.sqrt(np.mean(d[:, :, 0]**2 + d[:, :, 1]**2)),
                 'rms_velocity_m_s': np.sqrt(np.mean(d[:, :, 3]**2 + d[:, :, 4]**2)),
                 'rms_spin_rad_s': np.sqrt(np.mean(d[:, :, 5]**2))}
        for key, value in error.items(): assert np.isclose(value, comparison['errors'][key])
        within = all(error[k] <= limit for k, limit in plan['budget'].items())
        q = report['reference_qualification'][scene]['qualified']
        assert comparison['passed_qualified_reference'] == (q and within)
    for scene, qualification in report['reference_qualification'].items():
        passed = []
        for edge in qualification['refinement']:
            p, s = edge['from']; a = np.asarray(traces[f'{scene}__reference_p{p}_s{s}']['states'])
            p, s = edge['to']; b = np.asarray(traces[f'{scene}__reference_p{p}_s{s}']['states']); d = b - a
            metrics = [np.sqrt(np.mean(np.sum(d[:, :, :2]**2, axis=2))),
                       np.sqrt(np.mean(np.sum(d[:, :, 3:5]**2, axis=2))), np.sqrt(np.mean(d[:, :, 5]**2))]
            passed.append(all(x <= plan['reference_budget'][k] for x, k in zip(metrics, ('rms_position_m', 'rms_velocity_m_s', 'rms_spin_rad_s'))))
        assert qualification['qualified'] == all(passed)
    print('Audited 53 retained histories, all trajectory comparisons, analytic rows and reference gates.')


if __name__ == '__main__': audit()
