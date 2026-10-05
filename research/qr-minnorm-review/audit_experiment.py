"""Re-audit frozen QR receipts, original gates, fixed pairing and provenance."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from research.audit_large_contact_completion import check


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(phase):
    plan = json.loads((HERE / 'plan.json').read_text())
    build = json.loads((HERE / 'build-provenance.json').read_text())
    results = HERE / 'results'
    provenance = json.loads((results / (phase + '-provenance.json')).read_text())
    summary = json.loads((results / (phase + '-summary.json')).read_text())
    records = json.loads((results / (phase + '-receipts.json')).read_text())
    assert provenance['plan_sha256'] == build['plan_sha256'] == digest(HERE / 'plan.json')
    assert provenance['build_provenance_sha256'] == digest(HERE / 'build-provenance.json')
    assert provenance['runner_sha256'] == digest(HERE / 'run_experiment.py')
    committed = subprocess.check_output(['git', 'show', provenance['plan_commit'] + ':research/qr-minnorm-review/plan.json'], cwd=ROOT)
    assert committed == (HERE / 'plan.json').read_bytes()
    assert build['source_commit'] == plan['baseline_source_commit']
    for filename, expected in build['source_hashes'].items():
        raw = subprocess.check_output(['git', 'show', build['source_commit'] + ':' + filename], cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest() == expected
        assert digest(HERE / 'baseline' / Path(filename).name) == expected
    for variant in ['baseline', 'qr']:
        metadata = build['variants'][variant]
        assert digest(HERE / variant / 'replay') == metadata['binary_sha256']
        for filename, expected in metadata['files'].items():
            assert digest(HERE / variant / filename) == expected
    cases = {case['path']: case for case in plan['cases']}
    for case in cases.values():
        assert digest(ROOT / case['path']) == case['sha256']

    expected_order = []
    rng = random.Random(730142)
    if phase == 'validation':
        for case in plan['cases']:
            for order, variant in enumerate(['baseline', 'qr']):
                expected_order.append((case['path'], variant, 'untimed-validation', 0, order))
    else:
        for case in plan['cases']:
            variants = ['baseline', 'qr']
            rng.shuffle(variants)
            for order, variant in enumerate(variants):
                expected_order.append((case['path'], variant, 'untimed-warmup', 0, order))
        for repetition in range(plan['timing']['timed_pairs_per_capture']):
            ordered = plan['cases'].copy()
            rng.shuffle(ordered)
            for case in ordered:
                variants = ['baseline', 'qr']
                rng.shuffle(variants)
                for order, variant in enumerate(variants):
                    expected_order.append((case['path'], variant, 'timed', repetition, order))
    actual_order = [(r['capture'], r['variant'], r['kind'], r['repetition'], r['within_pair_order']) for r in records]
    assert actual_order == expected_order
    accepted = 0
    max_accepted_residual = 0.
    for row in records:
        receipt = row['receipt']
        assert row['capture_sha256'] == cases[row['capture']]['sha256']
        assert math.isfinite(receipt['solve_time_s']) and receipt['solve_time_s'] >= 0
        assert row['process_elapsed_s'] >= receipt['solve_time_s']
        data = json.loads((ROOT / row['capture']).read_text())
        assert receipt['iteration_budget'] == data['iteration_budget']
        assert receipt['tolerance_m_s'] == data['tolerance_m_s']
        recomputed = check(ROOT / row['capture'], receipt['p'])
        assert recomputed == row['independent_gate']
        assert bool(receipt['accepted']) == bool(receipt['solver_accepted'] and receipt['independent_law_accepted'] and receipt['independent_passivity_accepted'])
        if receipt['accepted']:
            assert recomputed['accepted'] and row['returncode'] == 0
            accepted += 1
            max_accepted_residual = max(max_accepted_residual, recomputed['independent_full_original_residual_m_s'])
        raw = np.asarray(receipt['p'])
        expected_w = np.asarray(data['A']) @ raw - np.asarray(data['b'])
        difference = float(np.max(np.abs(np.asarray(receipt['w']) - expected_w)))
        assert difference <= 1e-12 * max(1., float(np.max(np.abs(expected_w))))
    assert summary['complete'] and summary['attempt_count'] == len(records) and summary['accepted'] == accepted
    paired_cases = 0
    regressions = 0
    for item in summary['cases']:
        selected = [r for r in records if r['capture'] == item['capture'] and r['kind'] != 'untimed-warmup']
        for variant in ['baseline', 'qr']:
            chosen = [r for r in selected if r['variant'] == variant]
            assert item['variants'][variant]['attempts'] == len(chosen)
            assert item['variants'][variant]['accepted'] == sum(bool(r['receipt']['accepted']) for r in chosen)
            if phase == 'timing':
                times = [r['receipt']['solve_time_s'] for r in chosen]
                assert item['variants'][variant]['median_s'] == float(np.median(times))
                assert item['variants'][variant]['range_s'] == [min(times), max(times)]
        regression = item['variants']['baseline']['accepted'] > item['variants']['qr']['accepted']
        assert item['functional_regression'] == regression
        regressions += regression
        if 'paired_solve_cost_ratios_baseline_over_qr' in item:
            assert phase == 'timing'
            assert all(r['receipt']['accepted'] for r in selected)
            ratios = []
            for repetition in range(plan['timing']['timed_pairs_per_capture']):
                pair = {r['variant']: r['receipt']['solve_time_s'] for r in selected if r['repetition'] == repetition}
                ratios.append(pair['baseline'] / pair['qr'])
            assert item['paired_solve_cost_ratios_baseline_over_qr'] == ratios
            assert item['median_paired_solve_cost_ratio_baseline_over_qr'] == float(np.median(ratios))
            paired_cases += 1
        elif phase == 'timing':
            assert not all(r['receipt']['accepted'] for r in selected)
    assert summary['functional_regressions'] == regressions
    report = dict(schema='independent-frozen-qr-experiment-audit-v1', phase=phase,
                  accepted=True, attempts=len(records), accepted_attempts=accepted,
                  functional_regressions=regressions, successful_paired_cases=paired_cases,
                  maximum_accepted_original_residual_m_s=max_accepted_residual,
                  plan_commit=provenance['plan_commit'], source_commit=build['source_commit'],
                  receipt_sha256=digest(results / (phase + '-receipts.json')),
                  summary_sha256=digest(results / (phase + '-summary.json')),
                  auditor_sha256=digest(Path(__file__)),
                  scope='Frozen captured systems and descriptive native cost only; no full-world qualification or isolated-machine performance claim.')
    (results / (phase + '-independent-audit.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['validation', 'timing'], required=True)
    audit(parser.parse_args().phase)
