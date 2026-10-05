from pathlib import Path
import subprocess,json,os,hashlib,concurrent.futures
p=Path(__file__).resolve().parent;r=Path.cwd();env=dict(os.environ);env.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
common=['g++','-std=c++17','-O2','-ffp-contract=off','-DBT_USE_DOUBLE_PRECISION','-DSPATIAL_LAPACK_RECOVERY=1','-I'+str(r/'build/bullet-inspect/src'),'-I'+str(r/'build/spatial/_deps/json-src/single_include'),'-I'+str(p/'native-snapshot'),'-I'+str(p)]
libs=[str(r/'build/spatial/_deps/bullet-build/src'/x) for x in ['BulletDynamics/libBulletDynamics.a','BulletCollision/libBulletCollision.a','LinearMath/libLinearMath.a']]+['/lib/x86_64-linux-gnu/liblapack.so.3','/lib/x86_64-linux-gnu/libblas.so.3']
protected=[Path('/usr/bin/g++'),*(Path(x) for x in libs),r/'build/spatial/spatial_runner',r/'build/spatial/spatial_coulomb_replay']
commands={'baseline':common+[str(p/'native-snapshot/coulomb_replay.cpp'),*libs,'-o',str(p/'replay_baseline')],'candidate':common+[str(p/'candidate_replay.cpp'),*libs,'-o',str(p/'replay_candidate')]}
receipt={'thread_environment':{k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},'before':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected},'commands':commands};(p/'23-compile-plan.json').write_text(json.dumps(receipt,indent=2)+'\n')
def compile(v):
 z=subprocess.run(commands[v],env=env,capture_output=True,text=True);(p/f'23-{v}-compile.stdout').write_text(z.stdout);(p/f'23-{v}-compile.stderr').write_text(z.stderr);return v,z.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(compile,commands))
receipt.update(exit_codes=results,after={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected},executables={v:hashlib.sha256((p/('replay_'+v)).read_bytes()).hexdigest() for v,c in results.items() if c==0});(p/'23-compile-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(results,receipt['before']==receipt['after']);raise SystemExit(any(results.values()))
