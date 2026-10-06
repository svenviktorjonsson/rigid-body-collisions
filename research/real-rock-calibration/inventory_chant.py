"""Inspect public reconstruction files; inferred energy parameters are not measured tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def inspect(path):
    data = np.genfromtxt(path, delimiter=',', names=True)
    data = np.atleast_1d(data)
    speed = data['v']
    valid_mass = np.isfinite(speed) & np.isfinite(data['Ekin']) & (speed > 1e-8)
    mass = 2 * data['Ekin'][valid_mass] / speed[valid_mass]**2
    linear = np.column_stack([data['v_x'], data['v_y'], data['v_z']])
    omega = np.column_stack([data['omega_x'], data['omega_y'], data['omega_z']])
    norm = np.linalg.norm(omega, axis=1)
    valid = np.isfinite(omega).all(axis=1) & np.isfinite(data['Erot'])
    unit_valid = valid & np.isfinite(data['omega']) & (norm > 1e-8)
    ratios = data['omega'][unit_valid] / norm[unit_valid]
    record = {
        'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'rows': len(data), 'finite_gyro_energy_rows': int(valid.sum()),
        'energy_implied_mass_kg_median': float(np.median(mass)) if len(mass) else None,
        'energy_implied_mass_range_kg': [float(mass.min()), float(mass.max())] if len(mass) else None,
        'linear_speed_vector_consistency_max_m_s': float(np.nanmax(abs(np.linalg.norm(linear, axis=1)-speed))),
        'resultant_to_component_omega_ratio_median': float(np.median(ratios)) if len(ratios) else None,
        'rad_to_degree_ratio_agrees_1e_minus_8': bool(len(ratios) and np.max(abs(ratios-180/np.pi)) < 1e-8),
        'inertia_status': 'inferred author energy formula; not independently measured',
        'full_3d_replay_qualified': False,
    }
    if valid.sum() >= 3:
        design = omega[valid]**2
        rhs = 2 * data['Erot'][valid]
        diagonal, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
        record['energy_diagonal_fit'] = {
            'values_kg_m2_if_components_rad_s': diagonal.tolist(), 'rank': int(rank),
            'max_energy_equation_residual_J': float(np.max(abs(design @ diagonal-rhs))),
            'coordinate_basis_verified': False,
        }
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    files = sorted(args.folder.glob('*.txt'))
    assert len(files) == 82, f'Expected downloaded 82-file archive, got {len(files)}'
    records = [inspect(path) for path in files]
    output = {
        'source': 'https://doi.org/10.16904/envidat.174',
        'specimens': 'reinforced concrete ideal EOTA shapes; not natural rocks',
        'file_count': len(files),
        'files_with_finite_gyro_energy': sum(r['finite_gyro_energy_rows'] > 0 for r in records),
        'unverified_inputs': ['sensor/body/world orientation transforms', 'full attitude',
                              'per-impact contact point and ground normal', 'density distribution',
                              'independent inertia measurement', 'soil deformation/contact law'],
        'friction_fit_performed': False, 'records': records,
    }
    Path(__file__).with_name('chant-inventory.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print({k:output[k] for k in ('file_count','files_with_finite_gyro_energy','friction_fit_performed')})
