"""Publish-first bounded numerical-merit controls; all contact law gates exact."""
import argparse,hashlib,json,os,re,subprocess,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).parent;sha=lambda b:hashlib.sha256(b).hexdigest()
def check(data,raw):
 A=np.array(data['A']);b=np.array(data['b']);p=np.array(raw);hi=np.array(data['hi']);dep=np.array(data['dependencies']);w=A@p-b;res=0.;cone=0.
 for k in np.flatnonzero(dep<0):
  rows=np.flatnonzero(dep==k);eig=np.linalg.eigvalsh(A[np.ix_(rows,rows)])[-1];z=p[rows]-w[rows]/eig;length=np.linalg.norm(z);cap=hi[rows[0]]*max(0.,p[k]);proj=z if length<=cap else z*(cap/length)if length else z
  res=max(res,abs(p[k]-max(0.,p[k]-w[k]/A[k,k]))*A[k,k],np.linalg.norm(p[rows]-proj)*eig);cone=max(cone,np.linalg.norm(p[rows])-cap)
 ns=np.flatnonzero(dep<0);energy=float(.5*p@A@p-b@p);scale=float(1+np.sum(abs(p*b)));finite=bool(np.isfinite(p).all()and np.isfinite(w).all()and np.isfinite(energy)and np.isfinite(scale));tol=data['tolerance_m_s'];accepted=bool(finite and np.all(p[ns]>=0)and np.all(p[ns]<=hi[ns])and res<=tol and energy<=tol*scale)
 return dict(accepted=accepted,residual_m_s=float(res),passivity_J=energy,passivity_scale=scale,cone_excess_N_s=float(cone),finite=finite)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-commit',required=True);args=ap.parse_args();source=subprocess.check_output(['git','rev-parse',args.source_commit],text=True,cwd=ROOT).strip();plan=json.loads((P/'plan.json').read_text());own=list(plan['source_sha256'])+[str((P/'plan.json').relative_to(ROOT))];inputs=plan['capture_sha256'];env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 for path,digest in {**plan['source_sha256'],**inputs}.items():assert sha((ROOT/path).read_bytes())==digest,path
 for path in own:assert (ROOT/path).read_bytes()==subprocess.check_output(['git','show',source+':'+path],cwd=ROOT)
 headers=ROOT/'build/bullet-inspect/src';jsonheaders=ROOT/'build/spatial/_deps/json-src/single_include';static=[ROOT/'build/spatial/_deps/bullet-build/src'/f for f in ['BulletDynamics/libBulletDynamics.a','BulletCollision/libBulletCollision.a','LinearMath/libLinearMath.a']];guardpaths=list((ROOT/'spatial_backend').glob('*.h'))+list(headers.rglob('*.h'))+list(jsonheaders.rglob('*.hpp'))+static+[ROOT/'build/spatial/spatial_runner',ROOT/'build/spatial/spatial_coulomb_replay']+[ROOT/f for f in own]+[ROOT/f for f in inputs];guards={str(p):sha(p.read_bytes())for p in guardpaths}
 def guard():
  for p,h in guards.items():assert sha(Path(p).read_bytes())==h,p
 D=P/'results';D.mkdir(exist_ok=False);commands=[]
 for filename in ['replay','controls']:
  cmd=['c++','-std=c++17','-O2','-Wall','-Wextra','-ffp-contract=off','-DBT_USE_DOUBLE_PRECISION','-I'+str(headers),'-I'+str(jsonheaders),str(P/(filename+'.cpp'))]+[str(p)for p in static]+['/lib/x86_64-linux-gnu/liblapack.so.3','/lib/x86_64-linux-gnu/libblas.so.3','-o',str(D/filename)];commands.append(cmd);guard();proc=subprocess.run(cmd,capture_output=True,text=True,env=env);(D/(filename+'-compile.json')).write_text(json.dumps(dict(command=cmd,exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr),indent=2)+'\n');guard()
  if proc.returncode:raise RuntimeError('Failed research compilation retained')
 binaries={str(D/f):sha((D/f).read_bytes())for f in ['replay','controls']};runtime={}
 for filename in ['replay','controls']:
  ldd=subprocess.check_output(['ldd',str(D/filename)],text=True)
  for f in re.findall(r'^\s*\S+\s+=>\s+(/\S+)',ldd,re.M):runtime[str(Path(f).resolve())]=sha(Path(f).read_bytes())
 (D/'provenance.json').write_text(json.dumps(dict(source_commit=source,guards=guards,binaries=binaries,runtime_libraries=runtime,compile_commands=commands,thread_environment={k:env[k]for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},compiler_version=subprocess.check_output(['c++','--version'],text=True)),indent=2)+'\n')
 with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for path in own:z.write(ROOT/path,path)
  for path in (ROOT/'spatial_backend').glob('*.h'):z.write(path,str(path.relative_to(ROOT)))
 proc=subprocess.run([str(D/'controls')],capture_output=True,text=True,env=env);(D/'controls.stdout.json').write_text(proc.stdout);(D/'controls.stderr').write_text(proc.stderr);guard();assert proc.returncode==0;control=json.loads(proc.stdout);assert control['passed']and len(control['controls'])==9
 trials=[]
 for label,scaling,projection in plan['variants']:
  path=next(iter(inputs));proc=subprocess.run([str(D/'replay'),str(ROOT/path),str(scaling),str(projection)],capture_output=True,text=True,env=env);(D/(label+'.stdout.json')).write_text(proc.stdout);(D/(label+'.stderr')).write_text(proc.stderr);guard();assert proc.returncode==0;native=json.loads(proc.stdout);independent=check(json.loads((ROOT/path).read_text()),native['returned_impulse']);candidate=check(json.loads((ROOT/path).read_text()),native['candidate_impulse']);assert native['svd_calls']<=1024 and native['iteration_steps']<=1024 and native['decline_preserves_input'];assert native['accepted']==independent['accepted'];record=dict(label=label,native=native,independent=independent,last_candidate=candidate);trials.append(record);(D/(label+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(label,'PASS'if native['accepted']else'REJECT',native['svd_calls'],candidate['residual_m_s'],flush=True)
 guard()
 for path,h in {**binaries,**runtime}.items():assert sha(Path(path).read_bytes())==h,path
 (D/'summary.json').write_text(json.dumps(dict(controls_passed=9,trials=trials,guards_unchanged=True,scope='Research numerical merit variations only; all trials retained, no trajectory qualification or timing ranking.'),indent=2)+'\n')
if __name__=='__main__':main()
