"""Closed-form opposing-face control. No native execution or solver search."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent


def main():
    h = .001
    slop = 1e-9
    gap = .001 + slop
    penetration = -.0005
    erp = .2
    velocity = .95
    mass = 1.
    A = np.array([[1., -1.], [-1., 1.]]) / mass
    u = np.array([velocity, -velocity])
    physical_target = np.array([0., -gap / h])
    physical_pressure = np.zeros(2)
    physical_slack = u - physical_target
    assert np.all(physical_slack >= 0)
    assert np.max(abs(physical_pressure * physical_slack)) == 0
    pose_target = np.array([erp * -penetration / h, -(gap - slop) / h])
    pose_pressure = np.array([mass * pose_target[0], 0.])
    a = A @ pose_pressure
    pose_slack = a - pose_target
    assert np.min(np.linalg.eigvalsh(A)) >= 0
    assert np.all(pose_pressure >= 0) and np.all(pose_slack >= -1e-15)
    assert np.max(abs(pose_pressure * pose_slack)) <= 1e-15
    projected = np.maximum(0., pose_pressure - pose_slack / np.diag(A))
    residual = float(np.max(abs(projected - pose_pressure) * np.diag(A)))
    assert residual <= 1e-15
    actual_distance = gap + h * (u[1] + a[1])
    assert actual_distance < 0
    remaining = gap - slop + h * u[1]
    remaining_bound = -remaining / h
    assert a[1] < remaining_bound
    # An unchanged left penetration target conflicts with this tighter right
    # bound, although accepted physical motion alone resolves the left overlap.
    maximum_x_push = -remaining_bound
    assert maximum_x_push < pose_target[0]
    assert penetration + h * u[0] > 0
    combined_target = np.array([pose_target[0], pose_target[1]]) - u
    assert np.all(combined_target < 0)  # zero numerical pressure suffices
    report = dict(schema='opposing-face-clearance-algebraic-control-v1',
                  execution_source_commit='108a9bb4c7899f75d760b27b179cc56557904a08',
                  native_execution=False, actual_study_containment_failure=False,
                  h_s=h, declared_slop_m=slop, start_right_distance_m=gap,
                  start_left_distance_m=penetration, split_erp=erp, mass_kg=mass,
                  prescribed_final_physical_x_velocity_m_s=velocity,
                  physical_normal_targets_m_s=physical_target.tolist(),
                  physical_pressure_kg_m_s=physical_pressure.tolist(), physical_slack_m_s=physical_slack.tolist(),
                  position_mobility_per_kg=A.tolist(), position_targets_m_s=pose_target.tolist(),
                  position_pressure_kg_m_s=pose_pressure.tolist(), position_normal_velocity_m_s=a.tolist(),
                  position_slack_m_s=pose_slack.tolist(), position_projection_residual_m_s=residual,
                  combined_right_distance_m=actual_distance, right_remaining_clearance_after_physical_m=remaining,
                  proposed_right_remaining_clearance_pose_bound_m_s=remaining_bound,
                  unchanged_left_required_x_push_m_s=float(pose_target[0]),
                  remaining_right_maximum_x_push_m_s=float(maximum_x_push),
                  unchanged_penetrating_target_with_tightened_right_bound_feasible=False,
                  optional_all_rows_combined_pose_targets_m_s=combined_target.tolist(),
                  optional_all_rows_combined_zero_push_feasible=True,
                  scope='Closed-form admissible independent physical/pose rows can violate their combined clearance. No captured actual final physical contact velocity is inferred.')
    report['producer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (DIRECTORY / 'algebraic-control.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
