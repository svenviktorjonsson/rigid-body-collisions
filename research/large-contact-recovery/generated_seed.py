"""Guarded native-generated seed pilots on unchanged captured systems."""
import ast,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'generated-seed-plan.json').read_text());D=H/'results-generated-seeds';D.mkdir(exist_ok=False)
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(H/'generated_seed_replay.cpp')==plan['driver_source_sha256'];assert all(sha(H/'prototype-right-cone'/p)==v for p,v in plan['prototype_hashes'].items())
exe=Path(plan['binary']);exe.parent.mkdir(exist_ok=False);base=ROOT/'build/spatial/_deps/bullet-build/src';libs=[base/'BulletDynamics/libBulletDynamics.a',base/'BulletCollision/libBulletCollision.a',base/'LinearMath/libLinearMath.a',Path('/usr/lib/x86_64-linux-gnu/liblapack.so.3'),Path('/usr/lib/x86_64-linux-gnu/libblas.so.3')]
command=['c++','-O3','-DNDEBUG','-DBT_USE_DOUBLE_PRECISION','-DSPATIAL_LAPACK_RECOVERY=1','-I/home/viktor/.cache/physics-relation-20261005/deps/_deps/bullet-src/src','-I/home/viktor/.cache/physics-relation-20261005/deps/_deps/json-src/include','-I'+str(H/'prototype-right-cone'),str(H/'generated_seed_replay.cpp'),'-o',str(exe),*[str(p) for p in libs]]
r=subprocess.run(command,capture_output=True,text=True);(D/'compile.stdout').write_text(r.stdout);(D/'compile.stderr').write_text(r.stderr);save(D/'compile.json',{'command':command,'exit':r.returncode,'libraries':{str(p):sha(p) for p in libs}});r.check_returncode();exe_sha=sha(exe)
assert os.environ['OMP_NUM_THREADS']=='1' and os.environ['OPENBLAS_NUM_THREADS']=='1'
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent','exec'),ns);records=[]
for index,(path,hash_) in enumerate(plan['inputs'].items()):
 inp=ROOT/path;assert sha(inp)==hash_;d=json.loads(inp.read_text())
 for budget in plan['budgets']:
  start=time.perf_counter();r=subprocess.run([str(exe),str(inp),str(budget)],capture_output=True,text=True);prefix=D/f'{index}-{budget}';prefix.with_suffix('.stdout.json').write_text(r.stdout);prefix.with_suffix('.stderr').write_text(r.stderr);o=json.loads(r.stdout);gate=ns['external'](d,o);assert not o['accepted'] or gate['accepted'];production=False
  if o['accepted']:
   candidate=prefix.with_suffix('.candidate.json');save(candidate,dict(d,p=o['p']));native=subprocess.run([str(ROOT/'build/spatial/spatial_coulomb_replay'),str(candidate),'0'],capture_output=True,text=True);prefix.with_suffix('.production.json').write_text(native.stdout);production=native.returncode==0 and json.loads(native.stdout)['accepted']
  record={'input':path,'budget_per_component_per_seed':budget,'accepted':bool(o['accepted'] and gate['accepted'] and production),'prototype_accepted':o['accepted'],'production_zero_accepted':production,'independent':gate,'native_stats':o['prototype_direct_search'],'elapsed_s_descriptive':time.perf_counter()-start};records.append(record);save(D/'progress.json',records);print(index,budget,'accepted',record['accepted'],'prototype',o['accepted'],'residual',gate['projection_m_s'],flush=True)
  if record['accepted']:break
assert sha(exe)==exe_sha;save(D/'summary.json',{'complete':True,'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'binary_sha256':exe_sha,'records':records,'scope':'Native-generated numerical seed instantaneous proof only; production source untouched, world/refinement/performance gates still required.'})
