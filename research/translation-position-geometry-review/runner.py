"""Prospective exact original7301/ref2 replay with separate geometry observation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from spatial_engine import run
from research.spatial_scenes import container
DIRECTORY = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit')
    parser.add_argument('--check-plan',action='store_true')
    parser.add_argument('--results-directory',type=Path,default=DIRECTORY/'results')
    args = parser.parse_args()
    plan = json.loads((DIRECTORY/'plan.json').read_text())
    baseline = ROOT/plan['baseline_study']
    original = json.loads((baseline/'plan.json').read_text())
    config = next(c for c in original['scenes'] if c['id'] == plan['scene'])
    scene, half = container(**{k: v for k, v in config.items() if k not in ('id', 'fractions')})
    assert dict(scene=scene, half=half) == json.loads((baseline/'results/scenes.json').read_text())[plan['scene']]
    assert plan['scene']=='fast_rotate_shake27_hulls7301' and plan['lane']=='reference_2' and plan['travel_fraction']==.015
    for p,expected in plan['source_sha256'].items():
        assert sha((ROOT/p).read_bytes())==expected,p
    if args.check_plan:
        print('Plan valid: original7301/ref2 scene/material/timestep; geometry observer only; no native execution');return
    if not args.source_commit:parser.error('--source-commit is mandatory')
    source = subprocess.check_output(['git', 'rev-parse', args.source_commit],cwd=ROOT,text=True).strip()
    paths = sorted(set([str(p.relative_to(ROOT)) for p in (ROOT/'spatial_backend').glob('*.h')] +
                       ['spatial_backend/runner.cpp', 'spatial_backend/CMakeLists.txt',
                        'spatial_engine.py', 'research/spatial_scenes.py',
                        'research/translation-position-geometry-review/plan.json',
                        'research/translation-position-geometry-review/runner.py',
                        'research/translation-position-geometry-review/analyze.py',
                        'research/translation-position-geometry-review/requery.cpp',
                        str((baseline/'plan.json').relative_to(ROOT)),
                        str((baseline/'results/scenes.json').relative_to(ROOT))]))
    binary = ROOT/'build/spatial/spatial_runner'
    linked = subprocess.check_output(['ldd', str(binary)], text=True)
    import re
    libraries = {str(Path(p).resolve()): sha(Path(p).read_bytes())
                 for p in re.findall(r'^\s*\S+\s+=>\s+(/\S+)', linked, re.M)}
    sources = {p: sha((ROOT/p).read_bytes()) for p in paths}
    binary_hash = sha(binary.read_bytes())

    def guard():
        for p in paths:
            assert (ROOT/p).read_bytes() == subprocess.check_output(['git', 'show', source+':'+p],cwd=ROOT), p
        assert sha(binary.read_bytes()) == binary_hash
        for p, expected in libraries.items():
            assert sha(Path(p).read_bytes()) == expected, p

    guard()
    results = args.results_directory.resolve()
    results.mkdir(parents=True,exist_ok=False)
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
                  scope='Single-lane diagnostic; geometry re-query and changed pose-policy qualification require separate analysis.')
    if dump.exists():
        captured = json.loads(dump.read_text())
        record.update(matrix_schema=captured['schema'], phase=captured['phase'],
                      rows=len(captured['b']), capture_sha256=sha(dump.read_bytes()))
    geometry=Path(str(dump)+'.geometry.json')
    record['geometry_snapshot_available']=geometry.exists()
    if geometry.exists():
        data=json.loads(geometry.read_text());record.update(geometry_sha256=sha(geometry.read_bytes()),geometry_rows=len(data['rows']),geometry_schema=data['schema'])
    record['original_capture_byte_identical']=dump.exists() and sha(dump.read_bytes())==plan['capture_sha256']
    (results/'receipt.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'result'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
