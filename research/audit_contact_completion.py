"""Independent frozen-system law checks; captured solves do not qualify trajectories."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'f2097b00e7e539f05b5c48daced46f2e58e122bf'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(capture, impulse, claimed_residual):
    A = np.asarray(capture['A']); b = np.asarray(capture['b'])
    p = np.asarray(impulse); hi = np.asarray(capture['hi'])
    dep = np.asarray(capture['dependencies']); tol = capture['tolerance_m_s']
    assert A.shape == (len(b), len(b)) and p.shape == b.shape
    assert np.isfinite(A).all() and np.isfinite(p).all()
    w = A @ p - b; residual = 0.; support = 0.; complementarity = 0.
    scale = max(1., np.max(np.abs(p)))
    for n in np.flatnonzero(dep < 0):
        t = np.flatnonzero(dep == n); assert len(t) == 2
        assert 0 <= p[n] <= hi[n] and hi[t[0]] == hi[t[1]]
        a = np.linalg.eigvalsh(A[np.ix_(t, t)])[-1]
        cap = hi[t[0]] * p[n]; z = p[t] - w[t] / a
        norm = np.linalg.norm(z)
        projected = z * (min(1., cap / norm) if norm else 1.)
        residual = max(residual, abs(p[n] - max(0., p[n] - w[n] / A[n, n])) * A[n, n],
                       np.linalg.norm(p[t] - projected) * a)
        assert w[n] >= -tol and np.linalg.norm(p[t]) - cap <= tol / a
        complementarity = max(complementarity, abs(p[n] * w[n]))
        support = max(support, abs(p[t] @ w[t] + cap * np.linalg.norm(w[t])))
    energy = .5 * p @ (A @ p - 2 * b)
    energy_scale = 1 + np.sum(np.abs(p * b))
    assert residual <= tol and complementarity <= tol * scale and support <= tol * scale
    assert np.isfinite(energy) and energy <= tol * energy_scale
    assert abs(residual - claimed_residual) <= 1e-11
    return dict(residual_m_s=float(residual), support_gap_J=float(support), passive_bound_J=float(energy))


def audit():
    directory = ROOT / 'research/coulomb-trust'
    manifest = json.loads((directory / 'artifact-hashes.json').read_text())
    for name, digest in manifest.items():
        data = subprocess.check_output(['git', 'show', f'{SOURCE}:{name}'], cwd=ROOT) if name.startswith('spatial_backend/') else (ROOT / name).read_bytes()
        assert sha(data) == digest, name
    review = ROOT / 'research/completion-review'
    for name, digest in json.loads((review / 'manifest.json').read_text())['files'].items():
        assert sha((review / name).read_bytes()) == digest
    provenance = json.loads((directory / 'clean-header-provenance.json').read_text())
    with zipfile.ZipFile(directory / 'clean-header-source.zip') as archive:
        for name, digest in provenance['source_sha256'].items():
            assert sha(archive.read(name)) == digest
            assert archive.read(name) == subprocess.check_output(['git', 'show', f'{SOURCE}:{name}'], cwd=ROOT)
    assert sha((directory / 'clean-header-source.zip').read_bytes()) == provenance['source_archive_sha256']
    assert sha((directory / 'native-clean-header-six.jsonl').read_bytes()) == provenance['receipt_sha256']
    records = []
    for line in (directory / 'native-clean-header-six.jsonl').read_text().splitlines():
        row = json.loads(line); raw = (ROOT / row['capture']).read_bytes()
        assert sha(raw) == row['capture_sha256']
        result = check(json.loads(raw), row['p'], row['residual'])
        assert row['accepted'] and row['svd_calls'] <= 256 and row['attempts'] <= 96
        records.append(result)
    assert len(records) == 6
    combined = json.loads((directory / 'combined-solver-replays.json').read_text())
    assert len(combined) == 11
    for row in combined:
        capture = json.loads((ROOT / row['input']).read_text()); receipt = row['result']
        # The full replay receipt includes its independently evaluated law fields.
        assert receipt['accepted']
        check(capture, receipt['p'], receipt['stats']['residual_m_s'])
    print('Contact completion audit PASS: six isolated and eleven combined original-law captured solves; frozen review hashes')


if __name__ == '__main__':
    audit()
