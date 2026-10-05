"""Compare physical responses of strict baseline and QR validation roots."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    source = HERE / 'results/validation-receipts.json'
    rows = json.loads(source.read_text())
    records = []
    for left, right in zip(rows[::2], rows[1::2]):
        assert left['capture'] == right['capture']
        if not (left['receipt']['accepted'] and right['receipt']['accepted']):
            continue
        data = json.loads((ROOT / left['capture']).read_text())
        delta = np.asarray(right['independent_gate']['p']) - np.asarray(left['independent_gate']['p'])
        response = np.asarray(data['A']) @ delta
        maximum = float(np.max(np.abs(response)))
        tolerance = data['tolerance_m_s']
        records.append(dict(capture=left['capture'], rows=len(delta),
                            maximum_impulse_difference_N_s=float(np.max(np.abs(delta))),
                            maximum_physical_relative_velocity_difference_m_s=maximum,
                            tolerance_m_s=tolerance,
                            distinct_physical_response_above_ten_tolerances=maximum > 10 * tolerance))
    report = dict(schema='accepted-qr-baseline-root-comparison-v1',
                  receipt_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  reviewer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  pairs=records, accepted_pair_count=len(records),
                  distinct_physical_responses=sum(r['distinct_physical_response_above_ten_tolerances'] for r in records),
                  scope='Retained accepted validation roots only. A times impulse difference is the physical relative contact response; no proof of uniqueness or explanation of full trajectory refinement error.')
    (HERE / 'results/accepted-root-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Accepted root pairs', len(records), 'distinct physical responses', report['distinct_physical_responses'])
    print('Maximum response difference', max(r['maximum_physical_relative_velocity_difference_m_s'] for r in records))


if __name__ == '__main__':
    main()
