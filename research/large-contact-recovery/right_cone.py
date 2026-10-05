"""Unmodified native exact-component search from saved null-traction seeds."""
import ast,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'right-cone-plan.json').read_text());D=H/'results-right-cone';D.mkdir(exist_ok=False)
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
exe=Path(plan['binary']);exe.parent.mkdir(exist_ok=False)
assert all(sha(H/'prototype-right-cone'/p)==v for p,v in plan['prototype_hashes'].items())
base=ROOT/'build/spatial/_deps/bullet-build/src'
command=['c++','-O3','-DNDEBUG','-DBT_USE_DOUBLE_PRECISION','-DSPATIAL_LAPACK_RECOVERY=1','-I/home/viktor/.cache/physics-relation-20261005/deps/_deps/bullet-src/src','-I/home/viktor/.cache/physics-relation-20261005/deps/_deps/json-src/include',str(H/'prototype-right-cone/component_replay.cpp'),'-o',str(exe),str(base/'BulletDynamics/libBulletDynamics.a'),str(base/'BulletCollision/libBulletCollision.a'),str(base/'LinearMath/libLinearMath.a'),'/usr/lib/x86_64-linux-gnu/liblapack.so.3','/usr/lib/x86_64-linux-gnu/libblas.so.3']
r=subprocess.run(command,capture_output=True,text=True);(D/'compile.stdout').write_text(r.stdout);(D/'compile.stderr').write_text(r.stderr);save(D/'compile.json',{'command':command,'exit':r.returncode});r.check_returncode();exe_sha=sha(exe);save(D/'binary.json',{'sha256':exe_sha,'path':str(exe)})
assert os.environ['OMP_NUM_THREADS']=='1' and os.environ['OPENBLAS_NUM_THREADS']=='1'
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent','exec'),ns)
assert sha(H/'results-null-seeds/summary.json')==plan['seed_archive_sha256']
prior=json.loads((H/'results-null-seeds/summary.json').read_text());records=[]
for index,old in enumerate(prior['records']):
 path=ROOT/old['input'];assert sha(path)==old['sha256'];d=json.loads(path.read_text());targeted=set()
 for attempt in old['attempts']:
  if attempt['seed']=='warm' or attempt['seed'] in targeted:continue
  targeted.add(attempt['seed']);p=np.array(d['p']);p[attempt['original_rows']]=attempt['seed_p'];seed=D/f'{index}-{attempt["seed"]}.seed.json';save(seed,dict(d,p=p.tolist()))
  for budget in plan['budgets']:
   started=time.perf_counter();r=subprocess.run([str(exe),str(seed),str(budget)],capture_output=True,text=True);prefix=D/f'{index}-{attempt["seed"]}-{budget}';prefix.with_suffix('.stdout.json').write_text(r.stdout);prefix.with_suffix('.stderr').write_text(r.stderr);o=json.loads(r.stdout);gate=ns['external'](d,o);assert not o['accepted'] or gate['accepted'];record={'input':old['input'],'seed':attempt['seed'],'budget_per_component':budget,'native_exit':r.returncode,'accepted':bool(o['accepted'] and gate['accepted'] and r.returncode==0),'independent':gate,'descriptive_elapsed_s':time.perf_counter()-started,'native_stats':o['prototype_direct_search']};records.append(record);save(D/'progress.json',records);print(index,attempt['seed'],budget,'accepted',record['accepted'],gate['projection_m_s'],flush=True)
   if record['accepted']:break
  if records[-1]['accepted']:break
assert sha(exe)==exe_sha;save(D/'summary.json',{'complete':True,'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'records':records,'scope':'Unchanged native numerical search on immutable original A/b/law; seed changes only. No world or performance qualification.'})
