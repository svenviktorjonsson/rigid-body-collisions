"""Standard-library-only verification, independent of all numerical solvers."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path


def verify(data, witness):
    if data['schema'] != 'normal-only-position-rejection-v1':
        raise ValueError('Unsupported capture')
    A = [[Fraction(v) for v in row] for row in data['A']]
    b = list(map(Fraction, data['b']))
    upper = list(map(Fraction, data['hi']))
    n = len(b)
    if len(A) != n or any(len(row) != n for row in A) or len(upper) != n:
        raise ValueError('Incompatible dimensions')
    if len(data['lo']) != n or any(v != 0 for v in data['lo']) or any(v < 0 for v in upper):
        raise ValueError('Nonnegative bounded impulses required')
    if any(A[i][i] <= 0 for i in range(n)):
        raise ValueError('Positive diagonal required by projection gate')
    support = witness['support']
    weights = [Fraction(int(v['numerator']), int(v['denominator'])) for v in witness['weights']]
    if len(weights) != len(support) or len(set(support)) != len(support):
        raise ValueError('Invalid witness support')
    if any(i < 0 or i >= n for i in support) or any(v < 0 for v in weights) or sum(weights) != 1:
        raise ValueError('Normalized nonnegative witness required')
    row = [sum((y * A[i][j] for i, y in zip(support, weights)), Fraction(0)) for j in range(n)]
    target = sum((y * b[i] for i, y in zip(support, weights)), Fraction(0))
    maximum_response = sum((max(v, Fraction(0)) * hi for v, hi in zip(row, upper)), Fraction(0))
    tolerance = Fraction(data['tolerance_m_s'])
    if tolerance <= 0:
        raise ValueError('Positive tolerance required')
    lower_bound = target - maximum_response
    # The original projection residual bounds every inward normal velocity:
    # r_i = |p_i-max(0,p_i-w_i/A_ii)| A_ii >= max(0,-w_i).
    # Thus passing requires y.(A p-b) >= -tolerance. But bounded p implies
    # y.A.p <= sum_j max((y.A)_j,0)*upper_j, contradicting that condition.
    return dict(certified=lower_bound > tolerance,
                rows=n, support=support,
                weighted_target_m_s=float(target),
                maximum_bounded_response_m_s=float(maximum_response),
                residual_lower_bound_m_s=float(lower_bound),
                tolerance_m_s=float(tolerance),
                strict_margin_m_s=float(lower_bound - tolerance),
                exact_lower_bound=dict(numerator=str(lower_bound.numerator), denominator=str(lower_bound.denominator)),
                exact_margin=dict(numerator=str((lower_bound-tolerance).numerator), denominator=str((lower_bound-tolerance).denominator)),
                positive_response_columns=[i for i, v in enumerate(row) if v > 0])


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    witness = json.loads((directory / 'witness.json').read_text())
    capture = root / witness['capture']
    if hashlib.sha256(capture.read_bytes()).hexdigest() != witness['capture_sha256']:
        raise ValueError('Capture hash mismatch')
    report = verify(json.loads(capture.read_text()), witness)
    if not report['certified']:
        raise RuntimeError('No strict certificate')
    report['capture_sha256'] = witness['capture_sha256']
    destination = directory / 'independent-verification.json'
    text = json.dumps(report, indent=2) + '\n'
    if destination.exists() and destination.read_text() != text:
        raise ValueError('Refusing to replace different evidence')
    destination.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
