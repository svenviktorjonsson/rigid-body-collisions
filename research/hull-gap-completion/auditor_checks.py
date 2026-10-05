"""Controls for gap audit bookkeeping and unchanged historical gates.

No physics simulation is executed. Receipts are written only to a temporary
directory; historical archives and saved independent receipts are untouched.
"""
import copy
import json
from pathlib import Path
import tempfile

from research.audit_hull_gap_completion import FIELDS, ROOT, STUDY, audit_plan, audit_ledgers, ledger, position_projection
from research.audit_shared_hulls import audit


def rejects(call):
    try:
        call()
    except (AssertionError, KeyError, ValueError, TypeError):
        return
    raise AssertionError('Malformed evidence was accepted')


def main():
    controls = []
    record = dict(zip(FIELDS, [2, .01, -3., 5., [1., -2., 2.], 4.]))
    assert ledger(record) == record
    for key, value in ((FIELDS[0], True), (FIELDS[1], -1), (FIELDS[2], float('nan')),
                       (FIELDS[3], 2), (FIELDS[4], [1, 2]), (FIELDS[5], 2)):
        bad = dict(record)
        bad[key] = value
        rejects(lambda: ledger(bad))
    bad = dict(record)
    del bad[FIELDS[0]]
    rejects(lambda: ledger(bad))
    zero = dict(zip(FIELDS, [0, 0., 0., 0., [0., 0., 0.], 0.]))
    assert ledger(zero) == zero
    zero[FIELDS[1]] = .01
    rejects(lambda: ledger(zero))
    controls.append('finite/count/vector/triangle/zero-update ledger controls PASS')
    assert position_projection(dict(translation_split_solves=3, translation_split_residual_max_m_s=1e-9), 1e-8)['strict_position_projection_gate_passed']
    for residual in (float('nan'), float('inf'), -1., 1.001e-8):
        rejects(lambda: position_projection(dict(translation_split_solves=3, translation_split_residual_max_m_s=residual), 1e-8))
    controls.append('finite/nonnegative unchanged strict position projection residual controls PASS')
    original = audit_plan(STUDY)
    with tempfile.TemporaryDirectory(prefix='gap-audit-checks-') as name:
        temporary = Path(name)
        for mutate in (
                lambda p: p['physical_gates'].update(energy_change_minus_boundary_work_J=2.),
                lambda p: p['trajectory_budget'].update(position_m=.01),
                lambda p: p['common'].update(contact_tolerance_m_s=1e-7),
                lambda p: p['scenes'][0].update(fractions=[.06, .03, .01]),
                lambda p: p['declared_numerical_change']['position_stabilization'].update(baseline='split')):
            bad = copy.deepcopy(original)
            mutate(bad)
            (temporary / 'plan.json').write_text(json.dumps(bad))
            rejects(lambda: audit_plan(temporary))
        controls.append('original energy/contact/refinement/declaration controls PASS')
        (temporary / 'plan.json').write_text(json.dumps(original))
        prefix_path = temporary / 'results/progress' / original['scenes'][0]['id'] / 'reference_0.json'
        prefix_path.parent.mkdir(parents=True)
        prefix_path.write_text(json.dumps(dict(record, complete=False, translation_split_solves=2,
                                               translation_split_residual_max_m_s=1e-9)))
        prefix_report = audit_ledgers(temporary, 'ledger-control-only')
        assert prefix_report['snapshot_count'] == 1
        assert prefix_report['trajectory_qualified'] is False
        assert prefix_report['cases'][0]['trajectory_qualified'] is False
        assert prefix_report['cases'][0]['prefix_observation_only'] is True
        rejects(lambda: audit_ledgers(temporary, 'ledger-control-only', require_all=True))
        controls.append('partial accounting never qualifies and final accounting requires all snapshots/checkpoints PASS')
        historical = [
            ('hull-active-completion', '95d224f1cf6388dfc7b58b2a7719ee96c2f9971f', None),
            ('hull-translation-completion', '52f7e6d244e92a8405134e0222ce14fc3eda0ef6', 'split_translation'),
        ]
        for directory, source, policy in historical:
            study = ROOT / 'research' / directory
            # Read the archived execution SHA rather than assuming a shorthand.
            source = json.loads((study / 'results/summary.json').read_text())['execution_source_commit']
            report = audit(study, source, position_stabilization=policy, output_path=temporary / (directory + '.json'))
            receipt = study / ('independent-audit.json' if policy is None else 'final-independent-audit.json')
            old = json.loads(receipt.read_text())
            assert (temporary / (directory + '.json')).read_bytes() == receipt.read_bytes()
            for key in ('attempt_count', 'history_count', 'rejection_count', 'scenes'):
                assert report[key] == old[key]
            if policy is not None:
                rejects(lambda: audit(study, source, output_path=temporary / 'undeclared.json'))
            controls.append(directory + ' original physical gates and both refinement edges unchanged PASS')
    print(json.dumps(dict(schema='gap-auditor-controls-v1', controls=controls,
                         native_execution=False, historical_evidence_mutated=False), indent=2))


if __name__ == '__main__':
    main()
