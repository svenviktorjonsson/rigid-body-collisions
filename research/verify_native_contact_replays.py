"""Replay all 22 frozen contact systems and independently verify original gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from research.audit_large_contact_completion import check


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='build/spatial/spatial_coulomb_replay')
    parser.add_argument('--save', type=Path)
    parser.add_argument('--allow-rejections', action='store_true')
    args = parser.parse_args()
    plan = Path('research/qr-minnorm-review/plan.json')
    cases = json.loads(plan.read_text())['cases']
    assert len(cases) == 20 and len({c['sha256'] for c in cases}) == 20
    extension = Path('research/translation-native-review/v3/plan.json')
    cases = list(cases) + [dict(path=path, sha256=sha) for path, sha in
                           json.loads(extension.read_text())['captures'].items()]
    assert len(cases) == 22 and len({c['sha256'] for c in cases}) == 22
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1')
    sources = sorted(Path('spatial_backend').glob('*.h')) + [
        Path('spatial_backend/coulomb_replay.cpp'), Path('spatial_backend/runner.cpp'),
        Path('spatial_backend/CMakeLists.txt'), Path('spatial_engine.py'),
        Path(__file__).relative_to(Path.cwd())]
    initial_sources = {str(p): digest(p) for p in sources}
    initial_binary = digest(args.binary)
    records = []
    for case in cases:
        capture = Path(case['path'])
        assert digest(capture) == case['sha256'], capture
        start = time.perf_counter()
        process = subprocess.run([str(Path(args.binary).resolve()), str(capture)],
                                 capture_output=True, text=True, env=environment)
        native = json.loads(process.stdout)
        independent = check(capture, native['p'])
        assert native['accepted'] == independent['accepted'], capture
        assert process.returncode == (0 if native['accepted'] else 2), capture
        records.append(dict(capture=str(capture), capture_sha256=digest(capture),
                            process_elapsed_s=time.perf_counter()-start,
                            exit_code=process.returncode, native=native,
                            independent=independent))
        print(capture, 'PASS' if native['accepted'] else 'REJECT', flush=True)
    assert digest(args.binary) == initial_binary, 'Replay binary changed during study'
    assert {str(p): digest(p) for p in sources} == initial_sources, 'Source changed during study'
    libraries = {}
    linked = subprocess.check_output(['ldd', args.binary], text=True)
    for name, target in re.findall(r'^\s*(\S+)\s+=>\s+(/\S+)', linked, re.M):
        actual = Path(target).resolve()
        libraries[name] = dict(path=str(actual), sha256=digest(actual))
    result = dict(schema='native-twenty-two-original-contact-replay-v2',
                  source_base_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256={str(p): digest(p) for p in sources},
                  binary_sha256=digest(args.binary), linked_libraries=libraries,
                  plan_sha256=digest(plan), extension_plan_sha256=digest(extension), total_count=len(records),
                  accepted_count=sum(r['native']['accepted'] for r in records),
                  scope='Captured contact systems only; no trajectory or material qualification.',
                  records=records)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        if args.save.exists():
            raise RuntimeError('Use a new receipt destination; preserve earlier attempts')
        args.save.write_text(json.dumps(result, indent=2)+'\n')
    if not args.allow_rejections:
        assert result['accepted_count'] == len(cases), result['accepted_count']
    print('Independent original contact/passivity gates:', result['accepted_count'], '/ 22')


if __name__ == '__main__':
    main()
