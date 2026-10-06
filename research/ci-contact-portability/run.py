"""Portable hosted/local bounded research controls; retain every real decline."""
import ast,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=Path(sys.argv[1]).resolve();D.mkdir(parents=True,exist_ok=False);plan=json.loads((H/'plan.json').read_text())
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(H/p)==v for p,v in plan['guards'].items())
assert os.environ['OMP_NUM_THREADS']=='1' and os.environ['OPENBLAS_NUM_THREADS']=='1'
cache={}
for line in (ROOT/'build/spatial/CMakeCache.txt').read_text().splitlines():
 if '=' in line and ':' in line:cache[line.split(':')[0]]=line.split('=',1)[1]
bullet=Path(cache['BULLET_PHYSICS_SOURCE_DIR']);js=Path(cache['nlohmann_json_SOURCE_DIR']);lib=ROOT/'build/spatial/_deps/bullet-build/src'
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent','exec'),ns)
sources=[H/'run.py',H/'plan.json',*[p for p in (ROOT/'spatial_backend').glob('*') if p.is_file()]];guards={str(p):sha(p) for p in sources};records=[]
save(D/'environment.json',{'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'compiler':subprocess.check_output(['c++','--version'],text=True),'system':subprocess.check_output(['uname','-a'],text=True),'cpu':subprocess.check_output(['lscpu'],text=True),'guards':guards,'search_source_guards':plan['guards']})
for variant in ['seed16','tight6']:
 T=D/variant;T.mkdir();exe=T/'replay';command=['c++','-std=c++17','-O3','-DNDEBUG','-DBT_USE_DOUBLE_PRECISION','-DSPATIAL_LAPACK_RECOVERY=1','-ffp-contract=off','-I'+str(ROOT/'spatial_backend'),'-I'+str(bullet/'src'),'-I'+str(js/'include'),str(H/variant/'replay.cpp'),'-o',str(exe),str(lib/'BulletDynamics/libBulletDynamics.a'),str(lib/'BulletCollision/libBulletCollision.a'),str(lib/'LinearMath/libLinearMath.a'),'-l:liblapack.so.3','-l:libblas.so.3'];r=subprocess.run(command,text=True,capture_output=True);(T/'compile.stdout').write_text(r.stdout);(T/'compile.stderr').write_text(r.stderr);save(T/'compile.json',{'command':command,'exit':r.returncode});r.check_returncode()
 libs=re.findall(r'(/\S+)\s+\(',subprocess.check_output(['ldd',str(exe)],text=True));save(T/'runtime.json',{'binary_sha256':sha(exe),'libraries':{str(Path(p).resolve()):sha(Path(p).resolve()) for p in libs}})
 for i,(path,digest) in enumerate(plan['inputs'].items()):
  inp=ROOT/path;assert sha(inp)==digest;d=json.loads(inp.read_text());start=time.perf_counter();r=subprocess.run([str(exe),str(inp)],capture_output=True,text=True);(T/f'{i}.stdout.json').write_text(r.stdout);(T/f'{i}.stderr').write_text(r.stderr);o=json.loads(r.stdout);gate=ns['external'](d,o);assert r.returncode in [0,2] and (r.returncode==0)==o['accepted'];assert not o['accepted'] or gate['accepted']
  record={'variant':variant,'input':path,'accepted':bool(o['accepted'] and gate['accepted']),'native_exit':r.returncode,'independent':gate,'policy':o['null_traction_seed_policy'],'elapsed_s_descriptive':time.perf_counter()-start};records.append(record);save(D/'progress.json',records);print(variant,len(d['b']),'accepted',record['accepted'],'residual',gate['projection_m_s'],flush=True)
assert all(sha(p)==v for p,v in guards.items())
save(D/'summary.json',{'complete':True,'guards_unchanged':True,'records':records,'scope':'Instantaneous hosted/local search-control evidence only; retaining declines does not relax required production test or qualify model/world/performance.'})
