"""Render every fixed capture's predeclared paired cost statistic."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = [
    'initial hull42', 'initial hull7301', 'later hull7301', 'followup hull42',
    'shared followup hull42 ref1', 'rank hull7301 ref0', 'rank hull7301 ref1',
    'rank hull7301 ref2', 'rank hull42 ref0', 'rank hull42 ref1', 'rank hull42 ref2',
    'completion hull7301 ref0', 'completion hull7301 ref2',
    'completion hull42 ref0 (position)', 'completion hull42 ref1',
    'completion hull42 ref2', 'active hull42 ref0', 'active hull42 ref1',
    'active hull42 ref2', 'active hull7301 ref0 (position)',
]


def main():
    result = json.loads((HERE / 'results/timing-summary.json').read_text())
    assert result['complete'] and len(result['cases']) == len(LABELS)
    assert json.loads((HERE / 'results/timing-independent-audit.json').read_text())['accepted']
    lines = ['# Complete captured-system paired cost results', '',
             'All 240 scheduled attempts are retained: 40 warmups and 200 timed solves. '
             'Each table entry contains five timed paired repetitions. Costs are native '
             'solver seconds, excluding parsing and process startup. The ratio is the '
             'predeclared median of paired baseline/QR costs; above one means lower QR '
             'cost in these observations. These are descriptive measurements under a '
             'shared workload, with no isolated-machine or whole-trajectory performance claim.', '',
             '| Capture | Rows | Baseline median [range], s | QR median [range], s | Median paired ratio | Accepted baseline / QR |',
             '|---|---:|---:|---:|---:|---:|']
    positive = negative = eligible = 0
    for label, case in zip(LABELS, result['cases']):
        left = case['variants']['baseline']; right = case['variants']['qr']
        def cost(entry):
            lo, hi = entry['range_s']
            return f"{entry['median_s']:.6g} [{lo:.6g}, {hi:.6g}]"
        ratio = case.get('median_paired_solve_cost_ratio_baseline_over_qr')
        if ratio is not None:
            eligible += 1
            positive += ratio > 1
            negative += ratio < 1
        rendered = f'{ratio:.4g}' if ratio is not None else 'excluded: failed'
        lines.append(f"| [{label}](../../{case['capture']}) | {case['rows']} | {cost(left)} | {cost(right)} | {rendered} | {left['accepted']}/5 / {right['accepted']}/5 |")
    lines += ['', f"There are {result['functional_regressions']} baseline-accepted regressions. "
              f'Of the {eligible} eligible captures, {positive} have a paired median ratio above '
              f'one and {negative} below one. This describes observed directions, without a '
              'confidence interval or generalization beyond these fixed captures. '
              'The three captures failed in both variants are retained, with no successful-cost ratio.', '',
              'QR proposes a numerical minimum-norm Newton direction; pressure and other required '
              'nullspace SVDs remain. Count reductions do not establish improvements by themselves. '
              'Source, input and binary hashes, random pair ordering, original iteration budgets, '
              'the full original law, finite/nonnegative checks and passivity were independently '
              're-audited. Numerical research here does not qualify a full physical trajectory.', '',
              'The frozen control source is 95d224f, preceding subsequent production repairs. '
              'The measured ratios cannot be transferred to that later implementation without '
              'another prospective comparison. The earlier v1 regression remains unchanged.', '',
              'Concurrency disclosure: per-attempt load averages and the 30-second process ledger '
              'are retained. A brief supplementary fixture compilation and audit execution '
              'occurred while timing was running. CPU4 affinity and single-thread settings '
              'do not establish machine isolation.', '']
    (HERE / 'cost-results.md').write_text('\n'.join(lines))
    print('Cost report includes', len(result['cases']), 'captures;', eligible, 'eligible ratios; regressions', result['functional_regressions'])


if __name__ == '__main__':
    main()
