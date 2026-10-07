"""Validate integrated recovery against the immutable 23-capture experiment."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT/'research/new-combined-contact-review/run-20261005T194211Z'
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
env = dict(os.environ)
env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
function = next(n for n in ast.parse((FROZEN/'run23.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name=='external')
namespace={'np':np}
exec(compile(ast.Module(body=[function],type_ignores=[]),'immutable-gate','exec'),namespace)
plan=json.loads((FROZEN/'23-capture-prospective-plan.json').read_text())
binary=ROOT/'build/spatial/spatial_coulomb_replay'
guardpaths=[binary,*[p for p in (ROOT/'spatial_backend').glob('*') if p.is_file()],ROOT/'spatial_engine.py']
guards={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in guardpaths}
records=[]
for i,(path,h) in enumerate(plan['corpus'].items()):
    raw=(ROOT/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==h,path
    data=json.loads(raw)
    command=[str(binary),str(ROOT/path),'4096']
    result=subprocess.run(command,capture_output=True,text=True,env=env)
    (OUT/f'{i:02d}.stdout.json').write_text(result.stdout)
    (OUT/f'{i:02d}.stderr').write_text(result.stderr)
    current=json.loads(result.stdout)
    baseline=json.loads((FROZEN/f'23-results/{i:02d}-baseline.stdout.json').read_text())
    checked=namespace['external'](data,current)
    preserved=all(np.asarray(current[k],dtype=np.float64).tobytes()==np.asarray(baseline[k],dtype=np.float64).tobytes() for k in ('p','w')) and current['stats']==baseline['stats']
    policy=current['projection_tail_policy']
    passed=result.returncode==0 and current['accepted'] and checked['accepted']
    passed &= policy['attempts']==policy['solves']+policy['declines'] and policy['svd_calls']<=2048*policy['attempts'] and policy['iteration_steps']<=2048*policy['attempts']
    passed &= (preserved and policy['attempts']==0) if i<22 else (policy['solves']==1 and not baseline['accepted'])
    record={'input':path,'input_sha256':h,'command':command,'exit':result.returncode,'independent':checked,'old_endpoint_bytes_and_counters_exact':preserved,'policy':policy,'passed':bool(passed)}
    records.append(record);(OUT/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
    print(i,'PASS' if passed else 'FAIL',flush=True)
unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in guards.items())
receipt={'passed':unchanged and len(records)==23 and all(r['passed'] for r in records),'guards':guards,'unchanged':unchanged,'records':records}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(not receipt['passed'])
