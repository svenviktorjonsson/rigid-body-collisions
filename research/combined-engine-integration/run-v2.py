"""Retained two-lane actual-world control. No source/build/test mutations."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np
from spatial_engine import run

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan = json.loads((DIRECTORY / 'plan.json').read_text())
    source = plan['production_source_commit']
    assert re.fullmatch(r'[0-9a-f]{40}', source)
    sources = ['spatial_engine.py', 'spatial_backend/runner.cpp', 'spatial_backend/CMakeLists.txt']
    sources += sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'spatial_backend').glob('*.h'))
    source_hashes = {}
    for path in sources:
        raw = (ROOT / path).read_bytes()
        assert raw == subprocess.check_output(['git', 'show', source + ':' + path], cwd=ROOT)
        source_hashes[path] = hashlib.sha256(raw).hexdigest()
    binary = ROOT / 'build/spatial/spatial_runner'
    binary_hash = sha(binary)
    linked = subprocess.check_output(['ldd', str(binary)], text=True)
    libraries = {str(Path(p).resolve()): sha(Path(p).resolve()) for p in re.findall(r'(/\S+)\s+\(', linked)}
    def guard():
        assert sha(binary) == binary_hash
        for path, expected in source_hashes.items():
            assert sha(ROOT / path) == expected
        for path, expected in libraries.items():
            assert sha(path) == expected
    output = DIRECTORY / plan['results_directory']
    output.mkdir(exist_ok=False)
    provenance = dict(schema='independent-combined-native-world-provenance-v1', production_source_commit=source,
                      production_source_hashes=source_hashes, binary_sha256=binary_hash,
                      runtime_library_hashes=libraries, plan_sha256=sha(DIRECTORY / 'plan.json'),
                      independent_launcher_sha256=sha(__file__), launcher_committed_at_production_source=False,
                      launcher_scope='Research plan/launcher authored before these runs; compiled production source bytes are committed at declared bca source. No claim that this later research launcher belongs to that earlier commit.',
                      performance_comparison=False)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    with zipfile.ZipFile(output / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(ROOT / path, path)
        for name in ('plan.json', 'run.py'):
            archive.write(DIRECTORY / name, 'independent-research/' + name)
    metrics = {}
    results = {}
    for policy in plan['policies_in_order']:
        guard()
        try:
            result = run(plan['scene'], **plan['common'], position_stabilization=policy,
                         rejected_contact_path=output / (policy + '-rejection.json'),
                         progress_checkpoint_path=output / (policy + '-progress.json'))
        except Exception as error:
            retained = dict(exception_type=type(error).__name__, reason=str(error), failed_policy=policy)
            if isinstance(error, subprocess.CalledProcessError):
                retained.update(exit_code=error.returncode, stdout=error.stdout, stderr=error.stderr)
            (output / (policy + '-failure.json')).write_text(json.dumps(retained, indent=2) + '\n')
            raise
        guard()
        (output / (policy + '-result.json')).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        progress = json.loads((output / (policy + '-progress.json')).read_text())
        state = np.asarray(result['states'], dtype=float)
        assert state.shape == (2, 3, 13) and np.isfinite(state).all()
        assert result['collision_updates'] == 1 and progress['complete'] and progress['completed_output_frames'] == progress['expected_output_frames'] == 1
        np.testing.assert_allclose(result['mass'], [0, 0, 1], rtol=0, atol=1e-12)
        inertia = np.asarray(result['inertia_body_kg_m2'][2])
        np.testing.assert_allclose(inertia, .004 * np.eye(3), rtol=0, atol=1e-12)
        np.testing.assert_allclose(state[:, 2, 7:13], [[.95, 0, 0, 0, 0, 0]] * 2, rtol=0, atol=1e-12)
        final = state[-1, 2]
        initial_K = .5 * result['mass'][2] * np.dot(state[0, 2, 7:10], state[0, 2, 7:10])
        final_K = .5 * result['mass'][2] * np.dot(final[7:10], final[7:10]) + .5 * final[10:13] @ inertia @ final[10:13]
        np.testing.assert_allclose([initial_K, final_K], [.45125, .45125], rtol=0, atol=1e-12)
        for key in ('coulomb_residual_max_m_s', 'translation_split_residual_max_m_s'):
            assert np.isfinite(result[key]) and 0 <= result[key] <= plan['common']['contact_tolerance_m_s']
            assert progress[key] == result[key]
        for key, value in result.items():
            if key.startswith('translation_pose_'):
                assert progress[key] == value
        assert result['translation_pose_potential_change_J'] == result['translation_pose_absolute_potential_change_J'] == 0
        assert result['translation_pose_orbital_change_kg_m2_s'] == [0, 0, 0]
        assert result['translation_pose_absolute_orbital_change_kg_m2_s'] == 0
        assert result['numerical_model']['position_stabilization'] == policy
        assert result['numerical_model']['early_component_recovery'] is False and result['early_component_policy']['attempts'] == 0
        expected_x = plan['geometric_targets']['old_expected_x_m' if policy == 'split_translation_gap' else 'new_expected_x_m']
        np.testing.assert_allclose(final[:3], [expected_x, 0, 0], rtol=0, atol=plan['geometric_targets']['coordinate_tolerance_m'])
        gap = plan['geometric_targets']['right_wall_interior_x_m'] - final[0] - .1
        if policy == 'split_translation_gap':
            assert gap < -4.9e-5
            np.testing.assert_allclose(result['translation_pose_displacement_max_m'], .0001, rtol=0, atol=1e-12)
        else:
            assert gap > 4.9e-5 and result['translation_pose_displacement_max_m'] == 0
        metrics[policy] = dict(final_x_m=float(final[0]), right_gap_m=float(gap), final_vx_m_s=float(final[7]),
                              initial_kinetic_energy_J=float(initial_K), final_kinetic_energy_J=float(final_K),
                              native_physical_residual_max_m_s=result['coulomb_residual_max_m_s'],
                              native_position_residual_max_m_s=result['translation_split_residual_max_m_s'],
                              numerical_pose_displacement_max_m=result['translation_pose_displacement_max_m'])
        results[policy] = result
    assert len({r['physical_setup_id'] for r in results.values()}) == 1
    guard()
    report = dict(schema='independent-combined-native-world-control-v1', production_source_commit=source,
                  plan_sha256=provenance['plan_sha256'], all_prespecified_controls_passed=True, cases=metrics,
                  shared_physical_setup_id=next(iter(results.values()))['physical_setup_id'],
                  actual_old_policy_containment_defect_observed=True, defect_fixed_in_this_control=True,
                  production_sources_binary_runtime_unchanged=True, performance_comparison=False,
                  general_trajectory_accuracy_qualified=False, calibrated_friction_or_rubber_claim=False)
    (output / 'independent-audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
