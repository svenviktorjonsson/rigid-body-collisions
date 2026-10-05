"""Prospective isolated recovery, sequential compilation and guarded pilots."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
H=Path(__file__).resolve().parent
ROOT=H.parents[1]
plan=json.loads((H/'plan.json').read_text())
OUT=H/'results'
OUT.mkdir(exist_ok=False)
BUILD=Path('/home/viktor/.cache/physics-large-recovery-20261006')
BUILD.mkdir(exist_ok=False)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
assert os.environ['OPENBLAS_NUM_THREADS']=='1' and os.environ['OMP_NUM_THREADS']=='1'
assert all(digest(H/'prototype'/p)==h for p,h in plan['prototype_hashes'].items())
source=ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py'
function=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external')
ns={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),'independent-gate','exec'),ns)
base=ROOT/'build/spatial/_deps/bullet-build/src'
libs=[base/'BulletDynamics/libBulletDynamics.a',base/'BulletCollision/libBulletCollision.a',base/'LinearMath/libLinearMath.a',Path('/usr/lib/x86_64-linux-gnu/liblapack.so.3'),Path('/usr/lib/x86_64-linux-gnu/libblas.so.3')]
records=[];receipts=[]
for variant,file in [('direct','replay.cpp'),('components','component_replay.cpp')]:
    exe=BUILD/variant
    command=['c++','-O3','-DNDEBUG','-DBT_USE_DOUBLE_PRECISION','-DSPATIAL_LAPACK_RECOVERY=1','-I/home/viktor/.cache/physics-relation-20261005/deps/_deps/bullet-src/src','-I/home/viktor/.cache/physics-relation-20261005/deps/_deps/json-src/include',str(H/'prototype'/file),'-o',str(exe),*[str(p) for p in libs]]
    result=subprocess.run(command,capture_output=True,text=True)
    (OUT/f'{variant}-compile.stdout').write_text(result.stdout);(OUT/f'{variant}-compile.stderr').write_text(result.stderr)
    receipts.append({'command':command,'exit':result.returncode});save(OUT/'compilation.json',receipts)
    result.check_returncode()
    for index,(path,sha) in enumerate(plan['inputs'].items()):
        inp=ROOT/path;assert digest(inp)==sha;data=json.loads(inp.read_text())
        for budget in plan['budgets']:
            start=time.perf_counter();result=subprocess.run([str(exe),str(inp),str(budget)],capture_output=True,text=True);elapsed=time.perf_counter()-start
            prefix=OUT/f'{variant}-{index}-{budget}'
            prefix.with_suffix('.stdout.json').write_text(result.stdout);prefix.with_suffix('.stderr').write_text(result.stderr)
            output=json.loads(result.stdout);check=ns['external'](data,output)
            assert bool(output['accepted'])==bool(result.returncode==0)
            assert not output['accepted'] or check['accepted']
            stats=output['prototype_direct_search'];limit=budget*(output.get('exact_components',{}).get('attempts',1))
            assert stats['svd_calls']<=limit and stats['iteration_steps']<=limit
            record={'variant':variant,'input':path,'input_sha256':sha,'budget_per_component_or_direct':budget,'exit':result.returncode,'accepted':output['accepted'],'independent':check,'stats':stats,'component_policy':output.get('exact_components'),'elapsed_s_descriptive':elapsed,'binary_sha256':digest(exe)}
            records.append(record);save(OUT/'progress.json',records)
            print(variant,index,budget,'accepted',output['accepted'],'svds',stats['svd_calls'],flush=True)
            if output['accepted']:break
assert all(digest(H/'prototype'/p)==h for p,h in plan['prototype_hashes'].items())
save(OUT/'summary.json',{'complete':True,'prototype_source_unchanged':True,'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'records':records,'libraries':{str(p):digest(p) for p in libs},'scope':'Instantaneous-system evidence only; neither production nor trajectory acceptance.'})
