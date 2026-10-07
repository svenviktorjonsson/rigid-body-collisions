"""Read-only production receipt audit; never reruns or imports native solvers."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SOURCE = '9e97be07b833503c15d9a96f1f92afc59c11a292'
COLLATERAL = 'spatial_backend/position_geometry.h'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare_number(expected, recorded, label):
    require(isinstance(recorded, (int, float)) and math.isfinite(recorded), 'Nonfinite metric: ' + label)
    require(math.isclose(expected, recorded, rel_tol=1e-10, abs_tol=1e-10), 'Saved metric disagrees: ' + label)


def physical_gate(data, native):
    A = np.asarray(data['A'], dtype=float)
    b = np.asarray(data['b'], dtype=float)
    p = np.asarray(native['p'], dtype=float)
    dep = np.asarray(data['dependencies'], dtype=int)
    upper = np.asarray(data['hi'], dtype=float)
    lower = np.asarray(data['lo'], dtype=float)
    tolerance = data['tolerance_m_s']
    n = len(b)
    require(A.shape == (n, n) and p.shape == b.shape == dep.shape == upper.shape == lower.shape, 'Invalid recorded row shape')
    require(np.isfinite(A).all() and np.isfinite(b).all() and np.isfinite(p).all(), 'Nonfinite original rows/impulses')
    require(np.array_equal(A, A.T), 'Original mobility must be exactly symmetric')
    require(native['rows'] == n and native['tolerance_m_s'] == tolerance, 'Original row count/tolerance changed')
    require(native['iteration_budget'] == data.get('iteration_budget', 4096), 'Original iteration budget changed')
    w = A @ p - b
    np.testing.assert_allclose(native['w'], w, rtol=1e-12, atol=1e-12, err_msg='Native response is not original A*p-b')
    residual = 0.
    metrics = dict(negative_normal_velocity_m_s=0., negative_normal_impulse_scaled_m_s=0.,
                   normal_complementarity_work_J=0., active_normal_velocity_m_s=0.,
                   circle_violation_scaled_m_s=0., interior_sticking_velocity_m_s=0.,
                   maximum_dissipation_gap_J=0., positive_friction_work_J=0.)
    contacts = []
    covered = set()
    for k in np.flatnonzero(dep < 0):
        t = np.flatnonzero(dep == k)
        require(dep[k] == -1 and len(t) == 2 and lower[k] == 0 and A[k, k] > 0, 'Unsupported normal/contact dependency')
        require(0 <= p[k] <= upper[k], 'Original normal impulse bound failed')
        require(upper[t[0]] == upper[t[1]] >= 0 and np.array_equal(lower[t], -upper[t]), 'Original isotropic coefficient/bounds changed')
        T = A[np.ix_(t, t)]
        eigenvalues = np.linalg.eigvalsh(T)
        require(eigenvalues[0] > 0 and np.isfinite(eigenvalues).all(), 'Invalid tangent mobility')
        eigen = eigenvalues[-1]
        normal_projection = max(0., p[k] - w[k] / A[k, k])
        normal_error = abs(p[k] - normal_projection) * A[k, k]
        z = p[t] - w[t] / eigen
        cap = upper[t[0]] * p[k]
        length = np.linalg.norm(z)
        projection = z * min(1., cap / length) if length > 0 else z
        tangent_error = np.linalg.norm(p[t] - projection) * eigen
        require(np.isfinite([normal_error, tangent_error, cap]).all(), 'Nonfinite original projection metric')
        residual = max(residual, normal_error, tangent_error)
        pt = float(np.linalg.norm(p[t])); wt = float(np.linalg.norm(w[t])); work = float(p[t] @ w[t])
        metrics['negative_normal_velocity_m_s'] = max(metrics['negative_normal_velocity_m_s'], -w[k])
        metrics['negative_normal_impulse_scaled_m_s'] = max(metrics['negative_normal_impulse_scaled_m_s'], -p[k] * A[k, k])
        metrics['normal_complementarity_work_J'] = max(metrics['normal_complementarity_work_J'], abs(p[k] * w[k]))
        if p[k] * A[k, k] > tolerance:
            metrics['active_normal_velocity_m_s'] = max(metrics['active_normal_velocity_m_s'], abs(w[k]))
        metrics['circle_violation_scaled_m_s'] = max(metrics['circle_violation_scaled_m_s'], (pt - cap) * eigen)
        if cap - pt > tolerance / eigen:
            metrics['interior_sticking_velocity_m_s'] = max(metrics['interior_sticking_velocity_m_s'], wt)
        metrics['maximum_dissipation_gap_J'] = max(metrics['maximum_dissipation_gap_J'], abs(work + cap * wt))
        metrics['positive_friction_work_J'] = max(metrics['positive_friction_work_J'], work)
        contacts.append(dict(normal_row=int(k), normal_impulse=float(p[k]), normal_velocity_m_s=float(w[k]),
                             tangent_impulse_norm=pt, tangent_velocity_norm_m_s=wt,
                             friction_capacity=float(cap), friction_support_gap_J=work + cap * wt))
        covered.update([int(k), *map(int, t)])
    require(covered == set(range(n)), 'Orphan original contact rows')
    energy = float(.5 * p @ A @ p - b @ p)
    energy_scale = float(1 + np.sum(np.abs(p * b)))
    impulse_scale = max(1., float(np.max(np.abs(p))))
    require(math.isfinite(energy) and math.isfinite(energy_scale) and energy <= tolerance * energy_scale, 'Original finite passivity gate failed')
    require(residual <= tolerance, 'Original full-row projection gate failed')
    work_fields = {'normal_complementarity_work_J', 'maximum_dissipation_gap_J', 'positive_friction_work_J'}
    for name, value in metrics.items():
        require(value <= tolerance * (impulse_scale if name in work_fields else 1), 'Native physical law failed: ' + name)
        compare_number(value, native[name], name)
    compare_number(energy, native['passive_change_bound_J'], 'passive_change_bound_J')
    compare_number(energy_scale, native['passivity_scale'], 'passivity_scale')
    require(len(contacts) == len(native['contacts']), 'Saved native contact coverage changed')
    for expected, recorded in zip(contacts, native['contacts']):
        require(expected['normal_row'] == recorded['normal_row'], 'Saved native contact order changed')
        for field in expected.keys() - {'normal_row'}:
            compare_number(expected[field], recorded[field], field)
    require(all(native[name] is True for name in ['accepted', 'solver_accepted', 'independent_law_accepted', 'independent_passivity_accepted']), 'Native acceptance flags disagree')
    return dict(original_projection_residual_m_s=float(residual), passive_change_bound_J=energy,
                passivity_scale=energy_scale, native_physical_metrics={k: float(v) for k, v in metrics.items()},
                all_original_bounds_passed=True, all_original_rows_passed=True)


def audit(base=BASE, check_current_runtime=False):
    base = Path(base)
    receipt_path = base / 'twenty-two-replays.json'
    receipt = json.loads(receipt_path.read_text())
    plan = json.loads((base / 'plan.json').read_text())
    require(receipt['schema'] == 'native-twenty-two-original-contact-replay-v2' and receipt['source_base_commit'] == SOURCE, 'Unexpected execution identity')
    require(receipt['total_count'] == receipt['accepted_count'] == plan['expected_count'] == 22, 'Incomplete original corpus')
    initial = json.loads((ROOT / plan['capture_plan']).read_text())
    extension = json.loads((ROOT / plan['extension_plan']).read_text())
    require(digest(ROOT / plan['capture_plan']) == receipt['plan_sha256'], 'Capture plan hash changed')
    require(digest(ROOT / plan['extension_plan']) == receipt['extension_plan_sha256'], 'Extension plan hash changed')
    expected = [(case['path'], case['sha256']) for case in initial['cases']] + list(extension['captures'].items())
    require(len(expected) == 22 and len(set(path for path, _ in expected)) == 22, 'Capture corpus is not distinct')
    require([(r['capture'], r['capture_sha256']) for r in receipt['records']] == expected, 'Saved corpus/order/hash differs from declared plans')
    sources = receipt['source_sha256']
    with zipfile.ZipFile(base / 'execution-source.zip') as archive:
        require(set(archive.namelist()) == set(sources), 'Recorded source archive coverage differs')
        for path, expected_hash in sources.items():
            raw = archive.read(path)
            require(valid_digest(expected_hash) and hashlib.sha256(raw).hexdigest() == expected_hash, 'Archived source bytes changed: ' + path)
            if path == COLLATERAL:
                exists = subprocess.run(['git', 'cat-file', '-e', SOURCE + ':' + path], cwd=ROOT, capture_output=True)
                require(exists.returncode != 0, 'Collateral unexpectedly belongs to the committed model')
            else:
                committed = subprocess.check_output(['git', 'show', SOURCE + ':' + path], cwd=ROOT)
                require(raw == committed, 'Recorded source differs from committed model: ' + path)
                require(b'position_geometry.h' not in raw, 'Collateral unexpectedly referenced by compiled source')
        for path, expected_hash in plan['source_hashes'].items():
            committed = subprocess.check_output(['git', 'show', SOURCE + ':' + path], cwd=ROOT)
            require(hashlib.sha256(committed).hexdigest() == expected_hash, 'Production qualification plan source hash differs: ' + path)
    require(valid_digest(receipt['binary_sha256']), 'Invalid recorded binary hash')
    libraries = receipt['linked_libraries']
    require(libraries and 'liblapack.so.3' in libraries and 'libblas.so.3' in libraries, 'Missing recorded LAPACK/BLAS metadata')
    for metadata in libraries.values():
        require(Path(metadata['path']).is_absolute() and valid_digest(metadata['sha256']), 'Invalid recorded library metadata')
    if check_current_runtime:
        binary = ROOT / 'build/spatial/spatial_coulomb_replay'
        require(digest(binary) == receipt['binary_sha256'], 'Current binary differs from archived execution')
        linked = subprocess.check_output(['ldd', str(binary)], text=True)
        current = {name: str(Path(path).resolve()) for name, path in re.findall(r'^\s*(\S+)\s+=>\s+(/\S+)', linked, re.M)}
        require(current == {name: metadata['path'] for name, metadata in libraries.items()}, 'Current linked library set differs')
        for metadata in libraries.values():
            require(digest(metadata['path']) == metadata['sha256'], 'Current runtime library bytes differ')
    previous_path = ROOT / 'research/native-recovery-integration/twenty-replays.json'
    previous = json.loads(previous_path.read_text())
    require(previous['accepted_count'] == previous['total_count'] == 20, 'Previous accepted corpus incomplete')
    previous_rows = previous['records']
    records = []
    caps = dict(support_helper_calls=8, support_passes=8, support_largest_rows=64,
                support_svd_calls=1024, support_iteration_steps=16384,
                support_pressure_svd_calls=1024, support_pivot_calls=8,
                supplemental_svd_calls=1024, supplemental_iteration_steps=2048,
                supplemental_pressure_svd_calls=128, supplemental_pivot_calls=1,
                active_passes=8, active_svd_calls=256, active_pressure_svd_calls=128,
                continuation_svd_calls=256, polish_svd_calls=256, pressure_svd_calls=128,
                null_pressure_svd_calls=128)
    for index, record in enumerate(receipt['records']):
        capture = ROOT / record['capture']
        require(digest(capture) == record['capture_sha256'], 'Original captured system bytes changed')
        data = json.loads(capture.read_text()); native = record['native']
        require(record['exit_code'] == 0 and math.isfinite(record['process_elapsed_s']) and record['process_elapsed_s'] > 0, 'Native execution did not finish successfully')
        physical = physical_gate(data, native)
        stats = native['stats']
        require(stats['lapack_recovery_compiled'] is True, 'Supplemental runtime model not compiled')
        for name, maximum in caps.items():
            value = stats.get(name, 0)
            require(isinstance(value, int) and 0 <= value <= maximum, 'Declared search budget exceeded: ' + name)
        require(stats['iteration_sweeps_total'] <= native['iteration_budget'], 'Original sweep budget exceeded')
        preserved = None
        if index < 20:
            old = previous_rows[index]
            require(old['capture'] == record['capture'] and old['capture_sha256'] == record['capture_sha256'], 'Previous20 corpus differs')
            preserved = np.asarray(old['native']['p'], dtype='<f8').tobytes() == np.asarray(native['p'], dtype='<f8').tobytes()
            require(preserved, 'Old20 Float64 impulse bytes changed')
            require(stats['support_solves'] == 0, 'Old accepted system unexpectedly uses new tail')
        else:
            require(stats['support_solves'] == 1, 'New captured solution did not use the declared tail')
        records.append(dict(capture=record['capture'], capture_sha256=record['capture_sha256'],
                            original_row_count=len(data['b']), **physical,
                            old_float64_impulse_bytes_preserved=preserved,
                            recorded_budgets_passed=True, support_solves=stats['support_solves']))
    return dict(schema='independent-production-component-receipt-audit-v1', accepted=True,
                execution_source=SOURCE, count=22, accepted_count=22, old20_preserved=True,
                receipt_sha256=digest(receipt_path), source_archive_sha256=digest(base / 'execution-source.zip'),
                previous_receipt_sha256=digest(previous_path), qualification_plan_sha256=digest(base / 'plan.json'),
                auditor_sha256=digest(__file__), source_classification=dict(committed_model_source_count=len(sources) - 1,
                    archived_uncommitted_uncompiled_collateral={COLLATERAL: sources[COLLATERAL]}),
                recorded_runtime_metadata_valid=True, current_runtime_bytes_checked=check_current_runtime,
                recorded_budgets_checked=list(caps), records=records,
                limitations=['Pressure-attempt counts are not exported in the production replay; pressure SVD and guide-call counts are verified.',
                             'Guide internal pivot counts and total wall-clock work have no exposed hard bound.',
                             'Captured-system qualification does not qualify full trajectories, material calibration or performance superiority.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-current-runtime', action='store_true')
    parser.add_argument('--save', type=Path)
    args = parser.parse_args()
    result = audit(check_current_runtime=args.check_current_runtime)
    if args.save:
        require(not args.save.exists(), 'Preserve existing receipts; choose a new destination')
        args.save.write_text(json.dumps(result, indent=2) + '\n')
    print('Independent production receipt audit PASS:22 original systems; old20 Float64 impulses preserved; source/capture hashes and recorded budgets verified')
