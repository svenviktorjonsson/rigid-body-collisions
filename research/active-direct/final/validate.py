"""Build and audit exact captured systems; timings are descriptive only."""
from pathlib import Path
import hashlib,json,os,platform,subprocess,sys,time,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from research.coulomb_diagnostics import System
HERE=Path(__file__).resolve().parent
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 paths=json.loads((HERE.parent/'combined-capture-paths.json').read_text());assert len(paths)==20
 headers=sorted(HERE.rglob('*.h'));source_hashes={str(p.relative_to(ROOT)):sha(p) for p in headers}
 flags=['c++','-std=c++17','-O3','-DNDEBUG','-DBT_USE_DOUBLE_PRECISION','-ffp-contract=off','-Werror=misleading-indentation','-Wall','-Wextra','-I'+str(HERE),'-isystem',str(HERE/'frozen_baseline'),'-isystem',str(ROOT/'build/bullet-inspect/src'),'-isystem',str(ROOT/'build/spatial/_deps/json-src/include')]
 libs=[ROOT/'build/spatial/_deps/bullet-build/src'/part/('lib'+part+'.a') for part in ['BulletDynamics','BulletCollision','LinearMath']]+[Path('/lib/x86_64-linux-gnu/liblapack.so.3'),Path('/lib/x86_64-linux-gnu/libblas.so.3')]
 provenance={'schema':'bounded-contact-restart-provenance-v1','current_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'physical_input_scope':'Exact archived contact systems, not full trajectory qualification','source_hashes':source_hashes,'libraries':{str(p):{'resolved_path':str(p.resolve()),'sha256':sha(p)} for p in libs},'library_package_versions':subprocess.check_output(['dpkg-query','-W','liblapack3','libblas3'],text=True).strip(),'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],'python':sys.version,'numpy':np.__version__,'scipy':__import__('scipy').__version__,'platform':platform.platform(),'thread_environment':{k:ENV[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},'commands':[],'captures':{p:sha(ROOT/p) for p in paths}}
 for name in ['checks','replay']:
  binary=Path('/tmp')/('active-final-'+name+'-frozen');command=flags+[str(HERE/(name+'.cpp'))]+[str(p) for p in libs]+['-o',str(binary)];provenance['commands'].append(command);subprocess.run(command,cwd=ROOT,env=ENV,check=True);provenance[name+'_binary_sha256']=sha(binary)
 controls=subprocess.run(['/tmp/active-final-checks-frozen'],env=ENV,capture_output=True,text=True,check=True);(HERE/'controls.log').write_text(controls.stdout+controls.stderr)
 audits=[]
 with (HERE/'twenty-replays.jsonl').open('w') as output:
  for path in paths:
   started=time.perf_counter();result=subprocess.run(['/tmp/active-final-replay-frozen',path],cwd=ROOT,env=ENV,capture_output=True,text=True,check=True);r=json.loads(result.stdout);r['descriptive_wall_time_s']=time.perf_counter()-started;output.write(json.dumps(r)+'\n');output.flush()
   d=json.loads((ROOT/path).read_text());gate=System.from_dump(d).gate(np.array(r['p']),d['tolerance_m_s']);audits.append(dict(capture=path,capture_sha256=sha(ROOT/path),native_accepted=r['accepted'],independent_gate=gate,lane=r['lane'],supplement_svd_calls=r['supplement_svd_calls'],iteration_steps=r['iteration_steps']));print(path,r['accepted'],r['lane'],gate['residual_m_s'],flush=True)
 assert source_hashes=={str(p.relative_to(ROOT)):sha(p) for p in headers},'source changed during replay'
 assert provenance['captures']=={p:sha(ROOT/p) for p in paths},'captured inputs changed'
 all_accepted=all(a['native_accepted'] and a['independent_gate']['accepted'] for a in audits)
 (HERE/'independent-audit.json').write_text(json.dumps(dict(count=len(audits),all_accepted=all_accepted,results=audits),indent=2)+'\n')
 provenance['all_accepted']=all_accepted;(HERE/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 assert all_accepted
 print('PASS20 unchanged contact-system law/energy gates; full trajectories still require prospective runs',flush=True)
if __name__=='__main__':main()
