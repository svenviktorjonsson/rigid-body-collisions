"""Audit the explicitly prospective translation-only six-run hull protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from research.audit_shared_hulls import audit
from research.audit_hull_active_completion import audit_progress

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'research/hull-translation-completion'


def audit_runtime(study, source):
    """Verify the frozen runner's binary and recorded shared-library metadata.

    This checks the runner's declared ldd dependency set; it does not attest the
    dynamic loader, kernel, or an isolated execution environment.
    """
    study = Path(study)
    provenance_path = study / 'results/checkpoints/provenance.json'
    raw = provenance_path.read_bytes()
    provenance = json.loads(raw)
    assert provenance['execution_source_commit'] == source
    plan_path = study / 'plan.json'
    plan = json.loads(plan_path.read_text())
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert provenance['plan_sha256'] == digest(plan_path)
    committed = subprocess.check_output(['git', 'show', source + ':' + str(plan_path.relative_to(ROOT))], cwd=ROOT)
    assert committed == plan_path.read_bytes()
    assert plan['declared_numerical_change'] == {'position_stabilization': {'baseline': 'split', 'candidate': 'split_translation'}}
    assert plan['common']['position_stabilization'] == 'split_translation'
    binary = ROOT / 'build/spatial/spatial_runner'
    assert digest(binary) == provenance['binary_sha256']
    recorded = provenance['runtime_library_hashes']
    assert recorded and all(Path(path).is_absolute() for path in recorded)
    linked = subprocess.check_output(['ldd', str(binary)], text=True)
    resolved = {str(Path(path).resolve()) for path in re.findall(r'^\s*\S+\s+=>\s+(/\S+)', linked, re.M)}
    assert set(recorded) == resolved
    for path, expected in recorded.items():
        assert len(expected) == 64 and all(c in '0123456789abcdef' for c in expected)
        assert str(Path(path).resolve()) == path
        assert digest(path) == expected
    summary_path = study / 'results/summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for key in ('execution_source_commit', 'plan_sha256', 'binary_sha256', 'runtime_library_hashes', 'source_hashes'):
            assert summary[key] == provenance[key]
    report = dict(schema='independent-translation-runtime-provenance-audit-v1',
                  execution_source_commit=source, plan_sha256=provenance['plan_sha256'],
                  binary_sha256=provenance['binary_sha256'],
                  runtime_library_hashes=recorded, recorded_dependency_count=len(recorded),
                  provenance_sha256=hashlib.sha256(raw).hexdigest(),
                  auditor_sha256=digest(__file__), recorded_runtime_integrity_passed=True,
                  declared_numerical_change=plan['declared_numerical_change'],
                  trajectory_qualified=False,
                  scope='Current binary and all shared-library paths recorded by the frozen runner match their declared hashes. Dynamic loader, kernel and machine isolation are outside this metadata; runtime-only integrity does not qualify trajectories.')
    (study / 'independent-runtime-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Translation runtime provenance audit PASS:', len(recorded), 'recorded libraries; no trajectory qualification')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--runtime-only', action='store_true')
    args = parser.parse_args()
    source = subprocess.check_output(['git', 'rev-parse', args.source_commit],
                                     cwd=ROOT, text=True).strip()
    audit_runtime(STUDY, source)
    if not args.runtime_only:
        audit(STUDY, source, position_stabilization='split_translation')
        audit_progress(STUDY, source, require_all=True)
