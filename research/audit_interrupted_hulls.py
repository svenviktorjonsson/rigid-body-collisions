"""Audit retained interrupted work without treating interruption as completion."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'research/hull-completion'
SOURCE = 'f2097b00e7e539f05b5c48daced46f2e58e122bf'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit():
    results = DIRECTORY / 'results'; summary = json.loads((results / 'summary.json').read_text())
    plan = json.loads((DIRECTORY / 'plan.json').read_text())
    assert summary['execution_source_commit'] == SOURCE and not summary['complete']
    assert summary['attempt_count'] == summary['planned_attempt_count'] == 6
    assert summary['completed_attempt_count'] == 4 and summary['interrupted_attempt_count'] == 2
    assert summary['history_count'] == 0
    assert summary['plan_sha256'] == sha((DIRECTORY / 'plan.json').read_bytes())
    baseline = json.loads((ROOT / plan['baseline_plan']).read_text())
    for name in ['common', 'dt_s', 'trajectory_budget', 'physical_gates', 'scenes', 'reference_rule']:
        assert plan[name] == baseline[name]
    assert json.loads((results / 'scenes.json').read_text()) == json.loads((ROOT / plan['baseline_scenes']).read_text())
    for name, digest in summary['hashes'].items():
        assert sha((results / name).read_bytes()) == digest
    with zipfile.ZipFile(results / 'execution-source.zip') as archive:
        assert set(archive.namelist()) == set(summary['source_hashes'])
        for name, digest in summary['source_hashes'].items():
            raw = archive.read(name)
            assert sha(raw) == digest and raw == subprocess.check_output(['git', 'show', f'{SOURCE}:{name}'], cwd=ROOT)
    expected = {f"{scene['id']}/reference_{i}.json" for scene in plan['scenes'] for i in range(3)}
    completed = interrupted = captures = 0
    with zipfile.ZipFile(results / 'traces.zip') as archive:
        assert set(archive.namelist()) == expected
        for name in expected:
            row = json.loads(archive.read(name))
            if row['attempt_status'] == 'completed_rejection':
                completed += 1
                checkpoint = json.loads((results / 'checkpoints' / name).read_text())
                assert row == dict(checkpoint, attempt_status='completed_rejection')
                assert row['exit_code'] == 1 and row['elapsed_s'] > 0 and row['rejected']
            else:
                assert row['attempt_status'] == 'interrupted'; interrupted += 1
                assert row['interruption_reason'] and row['qualification'].startswith('unavailable')
                assert all(key not in row for key in ['exit_code', 'elapsed_s', 'states', 'rejected'])
            if row['rejection_dump'] is not None:
                captures += 1; raw = (results / row['rejection_dump']).read_bytes()
                receipt = summary['rejection_diagnostics'][name]; assert sha(raw) == receipt['sha256']
                data = json.loads(raw); A = np.asarray(data['A']); b = np.asarray(data['b'])
                assert A.shape == (len(b), len(b)) and np.isfinite(A).all() and np.isfinite(b).all()
                assert np.allclose(A, A.T, rtol=1e-12, atol=1e-12)
                assert data['residual_m_s'] > data['tolerance_m_s'] == plan['common']['contact_tolerance_m_s']
                assert receipt['phase'] == data['phase'] and receipt['rows'] == len(b)
    assert (completed, interrupted, captures) == (4, 2, 5)
    for receipt in summary['scenes'].values():
        assert not receipt['reference_qualified'] and receipt['choice'] is None
    print('Interrupted hull audit PASS: four completed rejections, two interruptions, five captures; zero qualification')


if __name__ == '__main__':
    audit()
