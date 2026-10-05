"""Recompute every frozen corpus gate and preservation check from archived outputs."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FROZEN = ROOT / 'research/new-combined-contact-review/run-20261005T194211Z'
tree = ast.parse((FROZEN / 'run23.py').read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'external')
namespace = {'np': np}
exec(compile(ast.Module(body=[function], type_ignores=[]), 'frozen-independent-gate', 'exec'), namespace)
with zipfile.ZipFile(HERE / 'results.zip') as archive:
    plan = json.loads(archive.read('23-capture-prospective-plan.json'))
    new_accepted = False
    for i, (path, expected) in enumerate(plan['corpus'].items()):
        raw = (ROOT / path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        data = json.loads(raw)
        outputs = [json.loads(archive.read(f'23-results/{i:02d}-{variant}.stdout.json')) for variant in ('baseline', 'candidate')]
        baseline, candidate = outputs
        assert candidate['accepted'] and namespace['external'](data, candidate)['accepted'], path
        if i < 22:
            assert baseline['accepted'] and namespace['external'](data, baseline)['accepted'], path
            assert all(baseline[k] == candidate[k] for k in ('p', 'w', 'stats')), path
            assert candidate['projection_policy']['attempts'] == 0, path
        else:
            assert not baseline['accepted']
            policy = candidate['projection_policy']
            assert policy['attempts'] == 1 and policy['solves'] == 1
            assert policy['guides'] <= 8 and policy['components'] <= 4
            assert policy['largest_rows'] <= 64 and policy['component_sweeps'] <= 16384
            new_accepted = True
    assert len(plan['corpus']) == 23 and new_accepted
    controls = json.loads(archive.read('controls-v2-result.json'))
    assert controls['exit'] == 0
    assert json.loads(archive.read('final-guards.json'))['unchanged']
print('23 original-law gates; 22 exact preserved responses; 9 rejection controls pass')
