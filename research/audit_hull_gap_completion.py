"""Independent, portable audit of the explicitly declared signed-gap study.

The original full-tensor physical gates and both quarter-budget refinement
edges remain unchanged. The pose ledger is numerical accounting, not an energy
correction to those gates. Native output prefixes never qualify trajectories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np

from research.audit_shared_hulls import audit as audit_archive
from research.audit_hull_active_completion import audit_progress

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'research/hull-gap-completion'
BASELINE_SOURCE = '52f7e6d244e92a8405134e0222ce14fc3eda0ef6'
CHANGE = {'position_stabilization': {'baseline': 'split_translation', 'candidate': 'split_translation_gap'}}
FIELDS = (
    'translation_pose_ledger_updates',
    'translation_pose_displacement_max_m',
    'translation_pose_potential_change_J',
    'translation_pose_absolute_potential_change_J',
    'translation_pose_orbital_change_kg_m2_s',
    'translation_pose_absolute_orbital_change_kg_m2_s',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def save(destination, report):
    """Every receipt is a new snapshot; never replace earlier evidence."""
    with Path(destination).open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


def audit_plan(study=STUDY):
    study = Path(study)
    plan = json.loads((study / 'plan.json').read_text())
    assert plan['baseline_source_commit'] == BASELINE_SOURCE
    assert plan['baseline_plan'] == 'research/hull-translation-completion/plan.json'
    assert plan['baseline_scenes'] == 'research/hull-translation-completion/results/scenes.json'
    baseline = json.loads((ROOT / plan['baseline_plan']).read_text())
    common = dict(baseline['common'], position_stabilization='split_translation_gap')
    assert plan['common'] == common and plan['declared_numerical_change'] == CHANGE
    for key in ('dt_s', 'trajectory_budget', 'physical_gates', 'reference_rule', 'scenes',
                'candidates', 'candidate_repetitions', 'contact_point_policy'):
        assert plan[key] == baseline[key]
    assert plan['candidates'] == {} and plan['candidate_repetitions'] == 0
    assert plan['native_ledger_fields'] == list(FIELDS)
    expected_paths = {
        plan['baseline_plan'], plan['baseline_scenes'],
        'research/hull-translation-completion/results/traces.zip',
        'research/hull-translation-completion/results/execution-source.zip',
    }
    assert set(plan['baseline_artifact_hashes']) == expected_paths
    for path, expected in plan['baseline_artifact_hashes'].items():
        assert is_digest(expected) and digest(ROOT / path) == expected
    return plan


def runtime_metadata(study, source):
    """Portable provenance/source checks; no live binary/library access."""
    study = Path(study)
    assert re.fullmatch(r'[0-9a-f]{40}', source)
    plan = audit_plan(study)
    path = study / 'results/checkpoints/provenance.json'
    provenance = json.loads(path.read_text())
    assert provenance['execution_source_commit'] == source
    assert provenance['plan_sha256'] == digest(study / 'plan.json')
    assert provenance['baseline_artifact_hashes'] == plan['baseline_artifact_hashes']
    assert is_digest(provenance['binary_sha256'])
    libraries = provenance['runtime_library_hashes']
    assert libraries and all(Path(p).is_absolute() and is_digest(h) for p, h in libraries.items())
    assert subprocess.check_output(['git', 'show', source + ':' + str((study / 'plan.json').relative_to(ROOT))], cwd=ROOT) == (study / 'plan.json').read_bytes()
    with zipfile.ZipFile(study / 'results/execution-source.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert set(archive.namelist()) == set(provenance['source_hashes'])
        for name, expected in provenance['source_hashes'].items():
            assert is_digest(expected)
            content = archive.read(name)
            assert hashlib.sha256(content).hexdigest() == expected
            assert content == subprocess.check_output(['git', 'show', source + ':' + name], cwd=ROOT)
        for name in ('research/audit_hull_gap_completion.py', 'research/audit_shared_hulls.py',
                     'research/audit_hull_active_completion.py'):
            assert name in provenance['source_hashes']
            # The executed auditor and its generic dependencies must themselves
            # match the frozen source, rather than silently using later code.
            assert (ROOT / name).read_bytes() == archive.read(name)
    summary_path = study / 'results/summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for key in ('execution_source_commit', 'plan_sha256', 'binary_sha256', 'runtime_library_hashes', 'source_hashes'):
            assert summary[key] == provenance[key]
    # A saved execution-host receipt, if present, is checked as archival evidence.
    # Its absence does not manufacture current-host attestation in portable CI.
    attestation_path = study / 'independent-current-runtime-audit.json'
    attestation_hash = None
    if attestation_path.exists():
        attestation = json.loads(attestation_path.read_text())
        assert attestation['schema'] == 'independent-gap-current-runtime-check-v1'
        for key in ('execution_source_commit', 'plan_sha256', 'binary_sha256', 'runtime_library_hashes'):
            assert attestation[key] == provenance[key]
        assert attestation['provenance_sha256'] == digest(path)
        assert attestation['auditor_sha256'] == provenance['source_hashes']['research/audit_hull_gap_completion.py']
        assert attestation['current_runtime_integrity_passed'] is True
        assert attestation['current_host_runtime_checked'] is True
        assert attestation['trajectory_qualified'] is False
        attestation_hash = digest(attestation_path)
    return provenance, plan, attestation_hash


def audit_runtime(study, source, output_path):
    provenance, _, _ = runtime_metadata(study, source)
    binary = ROOT / 'build/spatial/spatial_runner'
    assert digest(binary) == provenance['binary_sha256']
    linked = subprocess.check_output(['ldd', str(binary)], text=True)
    resolved = {str(Path(p).resolve()) for p in re.findall(r'^\s*\S+\s+=>\s+(/\S+)', linked, re.M)}
    libraries = provenance['runtime_library_hashes']
    assert set(libraries) == resolved
    for path, expected in libraries.items():
        assert str(Path(path).resolve()) == path and digest(path) == expected
    report = dict(schema='independent-gap-current-runtime-check-v1', execution_source_commit=source,
                  plan_sha256=provenance['plan_sha256'], binary_sha256=provenance['binary_sha256'],
                  runtime_library_hashes=libraries, provenance_sha256=digest(Path(study) / 'results/checkpoints/provenance.json'),
                  auditor_sha256=digest(__file__), current_host_runtime_checked=True,
                  current_runtime_integrity_passed=True, trajectory_qualified=False,
                  scope='Explicit local binary/library byte attestation. No trajectory qualification; portable archive checks do not require the current host to match.')
    save(output_path, report)
    return report


def ledger(record):
    """Validate finite accounting and signed-versus-absolute triangle bounds."""
    updates = record[FIELDS[0]]
    assert isinstance(updates, int) and not isinstance(updates, bool) and updates >= 0
    displacement, potential, absolute_potential = (record[key] for key in FIELDS[1:4])
    orbital = np.asarray(record[FIELDS[4]], dtype=float)
    absolute_orbital = record[FIELDS[5]]
    scalars = np.asarray([displacement, potential, absolute_potential, absolute_orbital], dtype=float)
    assert scalars.shape == (4,) and np.isfinite(scalars).all()
    assert orbital.shape == (3,) and np.isfinite(orbital).all()
    assert displacement >= 0 and absolute_potential >= 0 and absolute_orbital >= 0
    # Floating summation roundoff only; these checks do not loosen a physics gate.
    assert abs(potential) <= absolute_potential + 1e-12 * max(1., absolute_potential)
    orbital_norm = float(np.hypot.reduce(orbital))
    assert np.isfinite(orbital_norm)
    assert orbital_norm <= absolute_orbital + 1e-12 * max(1., absolute_orbital)
    if updates == 0:
        assert np.all(scalars == 0) and np.all(orbital == 0)
    return {key: record[key] for key in FIELDS}


def position_projection(record, tolerance):
    solves = record['translation_split_solves']
    residual = record['translation_split_residual_max_m_s']
    assert isinstance(solves, int) and not isinstance(solves, bool) and solves >= 0
    assert np.isfinite(residual) and 0 <= residual <= tolerance
    if solves == 0:
        assert residual == 0
    return dict(solves=solves, residual_max_m_s=residual, unchanged_tolerance_m_s=tolerance,
                strict_position_projection_gate_passed=True)


def audit_ledgers(study, source, require_all=False):
    study = Path(study)
    plan = audit_plan(study)
    summary_path = study / 'results/summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    cases = []
    for config in plan['scenes']:
        for i, fraction in enumerate(config['fractions']):
            name, lane = config['id'], f'reference_{i}'
            path = study / 'results/progress' / name / (lane + '.json')
            if not path.exists():
                assert not require_all
                continue
            raw = path.read_bytes()
            progress = json.loads(raw)
            prefix = ledger(progress)
            projection = position_projection(progress, plan['common']['contact_tolerance_m_s'])
            checkpoint = study / 'results/checkpoints' / name / (lane + '.json')
            final = None
            if require_all:
                assert checkpoint.exists()
            if checkpoint.exists():
                result = json.loads(checkpoint.read_text())
                same_snapshot = result['native_progress']['sha256'] == hashlib.sha256(raw).hexdigest()
                if require_all:
                    assert same_snapshot
                if same_snapshot and 'rejected' not in result:
                    assert progress['complete']
                    final = ledger(result)
                    assert final == prefix
                    assert position_projection(result, plan['common']['contact_tolerance_m_s']) == projection
                    assert result['numerical_model']['position_stabilization'] == 'split_translation_gap'
                    assert result['numerical_model']['position_clearance_target'] == 'negative available gap over internal timestep'
                saved_path = study / 'results/ledgers' / name / (lane + '.json')
                if require_all:
                    assert saved_path.exists() and summary is not None
                    saved_metadata = summary['native_ledger_receipts'][name + '/' + lane + '.json']
                    assert saved_metadata['path'] == str(saved_path.relative_to(study / 'results'))
                    assert saved_metadata['sha256'] == digest(saved_path)
                if same_snapshot and saved_path.exists():
                    saved = json.loads(saved_path.read_text())
                    assert saved['native_prefix'] == prefix and saved['prefix_available'] is True
                    assert saved['final'] == (final or {}) and saved['final_available'] == (final is not None)
            cases.append(dict(scene=name, lane=lane, fraction=fraction, progress_sha256=hashlib.sha256(raw).hexdigest(),
                              complete=progress['complete'], native_prefix=prefix, final=final,
                              position_projection=projection,
                              accounting_passed=True, prefix_observation_only=not progress['complete'], trajectory_qualified=False))
    return dict(schema='independent-gap-pose-ledger-audit-v1', execution_source_commit=source,
                snapshot_count=len(cases), cases=cases, accounting_passed=True, trajectory_qualified=False,
                scope='Finite native numerical pose accounting and signed/absolute bounds; final versus prefix consistency. Per-update impulses are not archived, so this is not independent reconstruction of each repair. Original endpoint physical gates remain uncorrected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=STUDY)
    parser.add_argument('--source-commit')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--runtime-metadata-only', action='store_true')
    parser.add_argument('--runtime-only', action='store_true')
    parser.add_argument('--check-current-runtime', action='store_true')
    parser.add_argument('--progress-only', action='store_true')
    parser.add_argument('--receipt-prefix', default='independent')
    args = parser.parse_args()
    study = args.directory.resolve()
    assert re.fullmatch(r'[a-zA-Z0-9_-]+', args.receipt_prefix)
    audit_plan(study)
    if args.plan_only:
        print('Gap study plan audit PASS; no execution or trajectory qualification')
        return
    if not args.source_commit:
        parser.error('--source-commit is required for execution evidence')
    source = subprocess.check_output(['git', 'rev-parse', args.source_commit], cwd=ROOT, text=True).strip()
    provenance, _, attestation = runtime_metadata(study, source)
    def destination(suffix):
        path = study / (args.receipt_prefix + suffix)
        assert not path.exists(), 'Choose a new receipt prefix to preserve earlier snapshots: ' + str(path)
        return path
    report = dict(schema='independent-gap-portable-runtime-metadata-audit-v1', execution_source_commit=source,
                  plan_sha256=provenance['plan_sha256'], binary_sha256=provenance['binary_sha256'],
                  runtime_library_hashes=provenance['runtime_library_hashes'],
                  provenance_sha256=digest(study / 'results/checkpoints/provenance.json'),
                  portable_metadata_integrity_passed=True, current_host_runtime_checked=False,
                  saved_local_attestation_sha256=attestation, declared_numerical_change=CHANGE, trajectory_qualified=False)
    save(destination('-runtime-metadata-audit.json'), report)
    if args.runtime_only or args.check_current_runtime:
        audit_runtime(study, source, destination('-current-runtime-audit.json'))
    if not args.runtime_only and not args.runtime_metadata_only:
        # Existing generic historical auditors retain their strict defaults.
        if not args.progress_only:
            audit_archive(study, source, position_stabilization='split_translation_gap', output_path=destination('-audit.json'))
        audit_progress(study, source, require_all=not args.progress_only, output_path=destination('-progress-audit.json'))
        save(destination('-pose-ledger-audit.json'), audit_ledgers(study, source, require_all=not args.progress_only))
    print('Gap archive audit PASS; partial prefixes are observations only')


if __name__ == '__main__':
    main()
