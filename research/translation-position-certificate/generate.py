"""Discover a numerical guide, then construct an exact bounded-row certificate.

The LP status and its approximate null vector never establish infeasibility.
Only verify.py's rational inequality has that authority for the frozen data.
"""
import hashlib
import json
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / 'research/translation-position-diagnostic/results/rejected-normal-system.json'


def eliminate(matrix):
    m = [row[:] for row in matrix]
    n = len(m)
    for k in range(n):
        pivot = next(i for i in range(k, n) if m[i][k])
        m[k], m[pivot] = m[pivot], m[k]
        value = m[k][k]
        m[k] = [v / value for v in m[k]]
        for i in range(n):
            if i != k:
                value = m[i][k]
                m[i] = [a - value * b for a, b in zip(m[i], m[k])]
    return [row[-1] for row in m]


def main():
    data = json.loads(CAPTURE.read_text())
    A = np.array(data['A'])
    b = np.array(data['b'])
    n = len(b)
    guide = linprog(-b, A_eq=np.vstack([A.T, np.ones(n)]),
                    b_eq=np.r_[np.zeros(n), 1], bounds=(0, None), method='highs',
                    options={'primal_feasibility_tolerance': 1e-7,
                             'dual_feasibility_tolerance': 1e-7})
    if not guide.success:
        raise RuntimeError('No discovery guide; no certificate claimed')
    # Discard tiny guide noise only when choosing a support to investigate.
    # Every original coefficient and row remains in the exact final check.
    support = np.flatnonzero(guide.x > 1e-4).tolist()
    exact = [[Q(value) for value in row] for row in data['A']]
    m = len(support)
    system = [[exact[i][j] for j in support] + [Q(-1), Q(0)] for i in support]
    system += [[Q(1)] * m + [Q(0), Q(1)]]
    weights = eliminate(system)[:m]
    if any(y < 0 for y in weights):
        raise RuntimeError('Discovery support has no nonnegative balanced witness')
    output = dict(schema='bounded-normal-row-certificate-v1',
                  capture=str(CAPTURE.relative_to(ROOT)),
                  capture_sha256=hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
                  arithmetic='Exact rational values of round-tripped IEEE binary64 inputs',
                  support=support,
                  weights=[dict(numerator=str(y.numerator), denominator=str(y.denominator)) for y in weights],
                  discovery=dict(method='HiGHS approximate normalized left-null LP',
                                 support_threshold=1e-4,
                                 approximate_weights=guide.x.tolist(),
                                 maximum_row_error=float(np.max(np.abs(guide.x @ A))),
                                 establishes_infeasibility=False))
    destination = Path(__file__).with_name('witness.json')
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
