"""Portable archive audit and opt-in local runtime checks for translation repair."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

from research.audit_shared_hulls import audit
from research.audit_hull_active_completion import audit_progress

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'research/hull-translation-completion'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def runtime_metadata(study, source):
    """Verify portable records without reading the current binary or libraries."""
    study = Path(study)
    provenance_path = study / 'results/checkpoints/provenance.json'
    raw = provenance_path.read_bytes()
    provenance = json.loads(raw)
    assert provenance['execution_source_commit'] == source
    plan_path = study / 'plan.json'
    plan = json.loads(plan_path.read_text())
    assert provenance['plan_sha256'] == digest(plan_path)
    committed = subprocess.check_output(['git', 'show', source + ':' + str(plan_path.relative_to(ROOT))], cwd=ROOT)
    assert committed == plan_path.read_bytes()
    assert plan['declared_numerical_change'] == {'position_stabilization': {'baseline': 'split', 'candidate': 'split_translation'}}
    assert plan['common']['position_stabilization'] == 'split_translation'
    assert is_digest(provenance['binary_sha256'])
    recorded = provenance['runtime_library_hashes']
    assert recorded and all(Path(path).is_absolute() and is_digest(value) for path, value in recorded.items())
    with zipfile.ZipFile(study / 'results/execution-source.zip') as archive:
        assert set(archive.namelist()) == set(provenance['source_hashes'])
        for path, expected in provenance['source_hashes'].items():
            assert is_digest(expected)
            content = archive.read(path)
            assert hashlib.sha256(content).hexdigest() == expected
            assert content == subprocess.check_output(['git', 'show', source + ':' + path], cwd=ROOT)
    summary_path = study / 'results/summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for key in ('execution_source_commit', 'plan_sha256', 'binary_sha256', 'runtime_library_hashes', 'source_hashes'):
            assert summary[key] == provenance[key]
    # Keep the original local attestation and its producer source immutable.
    attestation_path = study / 'independent-runtime-audit.json'
    attestation = json.loads(attestation_path.read_text())
    assert attestation['schema'] == 'independent-translation-runtime-provenance-audit-v1'
    for key in ('execution_source_commit', 'plan_sha256', 'binary_sha256', 'runtime_library_hashes'):
        assert attestation[key] == provenance[key]
    assert attestation['provenance_sha256'] == hashlib.sha256(raw).hexdigest()
    assert attestation['recorded_runtime_integrity_passed'] is True
    assert attestation['recorded_dependency_count'] == len(recorded)
    assert attestation['trajectory_qualified'] is False
    assert attestation['declared_numerical_change'] == plan['declared_numerical_change']
    assert attestation['auditor_sha256'] == digest(study / 'independent-runtime-audit-source.py')
    return provenance, plan, attestation_path


def audit_runtime_metadata(study, source, output_path=None):
    provenance, plan, attestation_path = runtime_metadata(study, source)
    report = dict(schema='independent-translation-portable-runtime-metadata-audit-v1',
                  execution_source_commit=source, plan_sha256=provenance['plan_sha256'],
                  binary_sha256=provenance['binary_sha256'],
                  runtime_library_hashes=provenance['runtime_library_hashes'],
                  local_attestation_sha256=digest(attestation_path),
                  local_attestation_source_sha256=digest(Path(study) / 'independent-runtime-audit-source.py'),
                  portable_metadata_integrity_passed=True, current_host_runtime_checked=False,
                  declared_numerical_change=plan['declared_numerical_change'], trajectory_qualified=False,
                  scope='Portable archive/schema/source/provenance checks and saved local attestation integrity. Current host binary/library equality is not required; no trajectory qualification.')
    destination = Path(output_path) if output_path is not None else Path(study) / 'independent-runtime-metadata-audit.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print('Portable runtime metadata audit PASS:', len(provenance['runtime_library_hashes']), 'recorded libraries; current host not checked')
    return report


def audit_runtime(study, source, output_path=None):
    """Explicit live check; preserve the original local attestation unchanged."""
    provenance, _, attestation_path = runtime_metadata(study, source)
    binary = ROOT / 'build/spatial/spatial_runner'
    assert digest(binary) == provenance['binary_sha256']
    linked = subprocess.check_output(['ldd', str(binary)], text=True)
    resolved = {str(Path(path).resolve()) for path in re.findall(r'^\s*\S+\s+=>\s+(/\S+)', linked, re.M)}
    recorded = provenance['runtime_library_hashes']
    assert set(recorded) == resolved
    for path, expected in recorded.items():
        assert str(Path(path).resolve()) == path and digest(path) == expected
    report = dict(schema='independent-translation-current-runtime-check-v1',
                  execution_source_commit=source, binary_sha256=provenance['binary_sha256'],
                  runtime_library_hashes=recorded, current_host_runtime_checked=True,
                  current_runtime_integrity_passed=True, trajectory_qualified=False,
                  original_local_attestation_sha256=digest(attestation_path),
                  auditor_sha256=digest(__file__),
                  scope='Explicit current-host binary and recorded-library byte check; original execution attestation remains unchanged. No trajectory qualification.')
    destination = Path(output_path) if output_path is not None else Path(study) / 'independent-current-runtime-audit.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print('Current local runtime audit PASS:', len(recorded), 'recorded libraries; original attestation preserved')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--runtime-only', action='store_true', help='Check current runtime, omit trajectory audit')
    parser.add_argument('--runtime-metadata-only', action='store_true', help='Portable metadata check only')
    parser.add_argument('--check-current-runtime', action='store_true')
    parser.add_argument('--receipt-prefix', default='independent', help='Use a new prefix to preserve earlier snapshot receipts')
    args = parser.parse_args()
    assert re.fullmatch(r'[a-zA-Z0-9_-]+', args.receipt_prefix)
    source = subprocess.check_output(['git', 'rev-parse', args.source_commit], cwd=ROOT, text=True).strip()
    audit_runtime_metadata(STUDY, source, STUDY / (args.receipt_prefix + '-runtime-metadata-audit.json'))
    if args.runtime_only or args.check_current_runtime:
        audit_runtime(STUDY, source, STUDY / (args.receipt_prefix + '-current-runtime-audit.json'))
    if not args.runtime_only and not args.runtime_metadata_only:
        audit(STUDY, source, position_stabilization='split_translation', output_path=STUDY / (args.receipt_prefix + '-audit.json'))
        audit_progress(STUDY, source, require_all=True, output_path=STUDY / (args.receipt_prefix + '-progress-audit.json'))
