"""Exact independent-mobility decomposition of retained circular-law captures.

Exploratory diagnostic, not a trajectory or a controlled cost experiment.
Only exactly zero off-component entries permit decomposition. Every normal and
both dependent tangent rows stay together; acceptance recomputes the original
full captured law and passivity independently of the native success flag.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.sparse.csgraph import connected_components
from research.audit_large_contact_completion import check

ROOT = Path(__file__).resolve().parents[1]


def partition(data):
    A = np.asarray(data['A'], dtype=float)
    dep = np.asarray(data['dependencies'], dtype=int)
    n = len(dep)
    if A.shape != (n, n) or not np.isfinite(A).all():
        raise ValueError('Finite square mobility required')
    adjacency = A != 0
    covered = np.zeros(n, dtype=bool)
    for k in np.flatnonzero(dep < 0):
        tangents = np.flatnonzero(dep == k)
        if len(tangents) != 2:
            raise ValueError('Two dependent tangent rows required')
        rows = np.r_[k, tangents]
        covered[rows] = True
        adjacency[np.ix_(rows, rows)] = True
    if not covered.all():
        raise ValueError('Orphan contact row')
    count, labels = connected_components(adjacency, directed=False)
    groups = [np.flatnonzero(labels == i) for i in range(count)]
    for rows in groups:
        others = np.flatnonzero(labels != labels[rows[0]])
        assert np.count_nonzero(A[np.ix_(rows, others)]) == 0
    return groups


def probe(capture, output):
    data = json.loads(capture.read_text())
    groups = partition(data)
    output.mkdir(parents=True, exist_ok=False)
    binary = ROOT / 'build/spatial/spatial_coulomb_replay'
    result = dict(schema='exact-coulomb-component-probe-v1',
                  capture=str(capture.relative_to(ROOT)),
                  capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
                  native_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  component_sizes=[len(rows) for rows in groups], trials=[],
                  scope='Exact original mobility components; descriptive concurrent timings only')
    full = np.zeros(len(data['b']))
    for i, rows in enumerate(groups):
        mapping = {int(old): new for new, old in enumerate(rows)}
        local = dict(data)
        local['A'] = np.asarray(data['A'])[np.ix_(rows, rows)].tolist()
        for key in ('b', 'p', 'lo', 'hi'):
            local[key] = np.asarray(data[key])[rows].tolist()
        local['dependencies'] = [-1 if data['dependencies'][k] < 0 else mapping[data['dependencies'][k]] for k in rows]
        path = output / f'component-{i}.json'
        path.write_text(json.dumps(local, allow_nan=False) + '\n')
        started = time.perf_counter()
        process = subprocess.run([str(binary), str(path)], text=True, capture_output=True)
        elapsed = time.perf_counter() - started
        (output / f'component-{i}-stdout.json').write_text(process.stdout)
        (output / f'component-{i}-stderr.txt').write_text(process.stderr)
        native = json.loads(process.stdout)
        full[rows] = native['p']
        result['trials'].append(dict(rows=rows.tolist(), exit_code=process.returncode,
                                     elapsed_s=elapsed, native_accepted=native['accepted'],
                                     independently_recomputed=check(path, native['p'])))
        print(capture.parent.name, capture.stem, 'component', i, len(rows), native['accepted'], flush=True)
    result['full_original_gate'] = check(capture, full)
    result['accepted'] = all(t['native_accepted'] for t in result['trials']) and result['full_original_gate']['accepted']
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('FULL ORIGINAL GATE', result['accepted'], result['full_original_gate']['independent_full_original_residual_m_s'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    probe(args.capture.resolve(), args.output.resolve())
