"""Reproduce one original lane with a normal-only position rejection observer."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

from spatial_engine import run
from research.spatial_scenes import container

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    source = subprocess.check_output(['git', 'rev-parse', args.source_commit], text=True).strip()
    plan = json.loads((DIRECTORY/'plan.json').read_text())
    baseline = ROOT/plan['baseline_study']
    original = json.loads((baseline/'plan.json').read_text())
    config = next(c for c in original['scenes'] if c['id'] == plan['scene'])
    scene, half = container(**{k: v for k, v in config.items() if k not in ('id', 'fractions')})
    assert dict(scene=scene, half=half) == json.loads((baseline/'results/scenes.json').read_text())[plan['scene']]
    paths = sorted(set([str(p.relative_to(ROOT)) for p in (ROOT/'spatial_backend').glob('*.h')] +
                       ['spatial_backend/runner.cpp', 'spatial_backend/CMakeLists.txt',
                        'spatial_engine.py', 'research/spatial_scenes.py',
                        'research/translation-position-diagnostic/plan.json',
                        'research/translation-position-diagnostic/runner.py']))
    binary = ROOT/'build/spatial/spatial_runner'
    linked = subprocess.check_output(['ldd', str(binary)], text=True)
    import re
    libraries = {str(Path(p).resolve()): sha(Path(p).read_bytes())
                 for p in re.findall(r'^\s*\S+\s+=>\s+(/\S+)', linked, re.M)}
    sources = {p: sha((ROOT/p).read_bytes()) for p in paths}
    binary_hash = sha(binary.read_bytes())

    def guard():
        for p in paths:
            assert (ROOT/p).read_bytes() == subprocess.check_output(['git', 'show', source+':'+p]), p
        assert sha(binary.read_bytes()) == binary_hash
        for p, expected in libraries.items():
            assert sha(Path(p).read_bytes()) == expected, p

    guard()
    results = DIRECTORY/'results'
    results.mkdir(exist_ok=False)
    with zipfile.ZipFile(results/'execution-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(ROOT/p, p)
    provenance = dict(execution_source_commit=source, source_hashes=sources,
                      binary_sha256=binary_hash, runtime_library_hashes=libraries,
                      plan_sha256=sha((DIRECTORY/'plan.json').read_bytes()))
    (results/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    dump = results/'rejected-normal-system.json'
    progress = results/'progress.json'
    start = time.perf_counter()
    try:
        result = run(scene, dt=original['dt_s'], travel_fraction=plan['travel_fraction'],
                     contact_point_policy=original['contact_point_policy'],
                     **original['common'], rejected_contact_path=dump,
                     progress_checkpoint_path=progress)
        record = dict(complete=True, result=result)
    except subprocess.CalledProcessError as error:
        record = dict(complete=False, exit_code=error.returncode,
                      error=error.stderr.strip())
    guard()
    record.update(elapsed_s=time.perf_counter()-start, matrix_snapshot_available=dump.exists(),
                  scope='Single-lane diagnostic; no trajectory or infeasibility qualification.')
    if dump.exists():
        captured = json.loads(dump.read_text())
        record.update(matrix_schema=captured['schema'], phase=captured['phase'],
                      rows=len(captured['b']), capture_sha256=sha(dump.read_bytes()))
    (results/'receipt.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'result'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
