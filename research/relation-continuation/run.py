"""Execute frozen relation proposal without changing production inputs."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'research/new-combined-contact-review/run-20261005T194211Z'
OUT = Path(sys.argv[1]).resolve()
DEPS = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
env = dict(os.environ)
env.update({k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
manifest = json.loads((SOURCE / 'relation-ready-manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected, name
guards = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'spatial_backend').glob('*') if p.is_file()}
provenance = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'guards': guards, 'proposal': manifest, 'commands': []}
libs = [DEPS / '_deps/bullet-build/src' / p for p in ('BulletDynamics/libBulletDynamics.a', 'BulletCollision/libBulletCollision.a', 'LinearMath/libLinearMath.a')]
for name, source in [('direct', SOURCE / 'relation_direct_replay.cpp'), ('candidate', SOURCE / 'relation_candidate_replay.cpp'), ('baseline', SOURCE / 'native-snapshot/coulomb_replay.cpp')]:
    command = ['c++', '-std=c++17', '-O2', '-ffp-contract=off', '-DBT_USE_DOUBLE_PRECISION', '-DSPATIAL_LAPACK_RECOVERY=1', '-I' + str(DEPS / '_deps/bullet-src/src'), '-I' + str(DEPS / '_deps/json-src/single_include'), '-I' + str(SOURCE / 'native-snapshot'), str(source), *map(str, libs), '/lib/x86_64-linux-gnu/liblapack.so.3', '/lib/x86_64-linux-gnu/libblas.so.3', '-o', str(OUT / ('replay_' + name))]
    provenance['commands'].append(command)
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    (OUT / (name + '-compile.json')).write_text(json.dumps({'command': command, 'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)
capture = ROOT / 'research/hull-combined-completion/results/rejections/fast_shake8_hulls42/reference_1.json'
result = subprocess.run([str(OUT / 'replay_direct'), str(capture), '4096'], capture_output=True, text=True, env=env)
(OUT / 'direct.stdout.json').write_text(result.stdout)
(OUT / 'direct.stderr').write_text(result.stderr)
(OUT / 'direct-exit.json').write_text(json.dumps({'exit': result.returncode}))
provenance['guard_check'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in guards.items())
(OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2))
print('Direct helper exit:', result.returncode, flush=True)
if result.returncode:
    raise SystemExit(result.returncode)
# Preserve the published auditor; use a separate launcher with explicit paths.
text = (SOURCE / 'run23.py').read_text()
text = text.replace("p=Path(__file__).resolve().parent;r=Path.cwd();", "p=Path(__file__).resolve().parent;r=Path(" + repr(str(ROOT)) + ");")
(OUT / '23-capture-prospective-plan.json').write_bytes((SOURCE / '23-capture-prospective-plan.json').read_bytes())
(OUT / 'run23.py').write_text(text)
raise SystemExit(subprocess.run([sys.executable, str(OUT / 'run23.py')], cwd=ROOT, env=env).returncode)
