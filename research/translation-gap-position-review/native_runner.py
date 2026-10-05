"""Prospective frozen-c465 native translation solver replay; no engine edits."""
import argparse,hashlib,json,re,subprocess,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).parent;sha=lambda data:hashlib.sha256(data).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-commit',required=True);a=ap.parse_args();source=subprocess.check_output(['git','rev-parse',a.source_commit],text=True,cwd=ROOT).strip();plan=json.loads((P/'native-plan.json').read_text());parent=json.loads((P/'plan.json').read_text());cap=ROOT/parent['capture'];geo=ROOT/parent['geometry'];paths=list(plan['source_sha256'])+[str((P/f).relative_to(ROOT))for f in ['native-plan.json','native_runner.py','plan.json']]
 def source_guard():
  for p in paths:assert (ROOT/p).read_bytes()==subprocess.check_output(['git','show',source+':'+p],cwd=ROOT),p
  for p,expected in plan['source_sha256'].items():assert sha((ROOT/p).read_bytes())==expected,p
  for p,origin in plan['frozen_header_origin'].items():assert (ROOT/p).read_bytes()==subprocess.check_output(['git','show',origin['source_commit']+':'+origin['original_path']],cwd=ROOT),p
  assert sha(cap.read_bytes())==parent['capture_sha256'] and sha(geo.read_bytes())==parent['geometry_sha256']
 source_guard();D=P/'native-results';D.mkdir(exist_ok=False);bullet=ROOT/'build/bullet-inspect/src';libs=[ROOT/'build/spatial/_deps/bullet-build/src'/v for v in ['BulletDynamics/libBulletDynamics.a','BulletCollision/libBulletCollision.a','LinearMath/libLinearMath.a']];commands=[]
 for name,cpp in [('native_replay',P/'native_replay.cpp'),('geometry_requery',ROOT/'research/translation-position-geometry-review/requery.cpp')]:
  cmd=['c++','-std=c++17','-O2','-DBT_USE_DOUBLE_PRECISION','-I'+str(bullet),'-I'+str(ROOT/'build/spatial/_deps/json-src/single_include'),str(cpp)]+[str(q)for q in libs]+['-o',str(D/name)];subprocess.run(cmd,check=True);commands.append(cmd)
 binaries={str(D/name):sha((D/name).read_bytes())for name in ['native_replay','geometry_requery']};libraries={str(p):sha(p.read_bytes())for p in libs}
 for name in ['native_replay','geometry_requery']:
  text=subprocess.check_output(['ldd',str(D/name)],text=True)
  for path in re.findall(r'^\s*\S+\s+=>\s+(/\S+)',text,re.M):libraries[str(Path(path).resolve())]=sha(Path(path).read_bytes())
 def guard():
  source_guard()
  for p,expected in {**binaries,**libraries}.items():assert sha(Path(p).read_bytes())==expected,p
 with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(ROOT/p,p)
  z.write(cap,parent['capture']);z.write(geo,parent['geometry'])
 provenance=dict(execution_source_commit=source,source_sha256={p:sha((ROOT/p).read_bytes())for p in paths},frozen_header_origin=plan['frozen_header_origin'],capture_sha256=parent['capture_sha256'],geometry_sha256=parent['geometry_sha256'],compiler_commands=commands,compiler_version=subprocess.check_output(['c++','--version'],text=True),binary_hashes=binaries,dependency_hashes=libraries)
 (D/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');guard()
 with open(D/'native.stdout','w')as log:r=subprocess.run([str(D/'native_replay'),str(cap),str(geo),str(D/'native-receipt.json')],stdout=log,stderr=subprocess.PIPE,text=True)
 guard();d=json.loads(cap.read_text());g=json.loads(geo.read_text());receipt=json.loads((D/'native-receipt.json').read_text());t=receipt['attempts'][0];A=np.array(d['A']);b=np.array(receipt['prospective_target']);p=np.array(t['impulse']);w=A@p-b;tol=d['tolerance_m_s'];res=float(np.max(abs(p-np.maximum(0,p-w/np.diag(A)))*np.diag(A)));energy=float(.5*p@A@p-b@p);scale=float(1+np.sum(abs(p*b)));bounds=bool(np.min(p)>=0 and np.all(p<=d['hi']));strict=bool(np.isfinite(p).all()and np.isfinite(w).all()and np.isfinite(energy)and np.isfinite(scale)and bounds and res<=tol and np.min(w)>=-tol and energy<=tol*scale)
 with open(D/'requery.stdout','w')as log:q=subprocess.run([str(D/'geometry_requery'),str(geo),str(D/'native-receipt.json'),str(D/'geometry-requery.json')],stdout=log,stderr=subprocess.PIPE,text=True)
 guard();query=json.loads((D/'geometry-requery.json').read_text())
 def pairmin(q):
  out={}
  for c in q['contacts']:
   pair=tuple(sorted([c['body_a'],c['body_b']]));out[pair]=min(out.get(pair,float('inf')),c['signed_distance_m'])
  return out
 before=pairmin(query['before']);after=pairmin(query['trials'][0]['after']);worsening=max([0.]+[min(before.get(k,0.),0.)-v for k,v in after.items()]);passed=bool(r.returncode==0 and q.returncode==0 and not receipt['original_native_solved']and receipt['native_solved']and strict and worsening<=parent['contact_slop_m']and t['max_translation_norm_m']<=parent['pose_bound_m'])
 report=dict(passed=passed,native_exit_code=r.returncode,native_stderr=r.stderr,geometry_exit_code=q.returncode,geometry_stderr=q.stderr,original_native_declined=not receipt['original_native_solved'],original_residual_m_s=receipt['original_native_residual_m_s'],independent_prospective_residual_m_s=res,independent_passive_bound_J=energy,bounds_passed=bounds,all74_independent_gates_passed=strict,largest_pair_gap_worsening_m=worsening,max_penetration_before_m=query['before']['max_penetration_m'],max_penetration_after_m=query['trials'][0]['after']['max_penetration_m'],scope='Prospective signed-gap position policy only. Original translation policy remains certified rejected; no full trajectory qualification.')
 (D/'independent-native-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));guard()
if __name__=='__main__':main()
