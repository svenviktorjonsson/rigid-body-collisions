"""Publish-first isolated native cache controls; production files stay untouched."""
import argparse,hashlib,json,re,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-commit',required=True);a=ap.parse_args();source=subprocess.check_output(['git','rev-parse',a.source_commit],text=True,cwd=ROOT).strip();plan=json.loads((P/'plan.json').read_text());paths=[str((P/f).relative_to(ROOT))for f in ['control.cpp','fixture.json','plan.json','runner.py']]
 def guard_source():
  for p in paths:assert (ROOT/p).read_bytes()==subprocess.check_output(['git','show',source+':'+p],cwd=ROOT),p
  for p,h in plan['source_sha256'].items():assert sha((ROOT/p).read_bytes())==h,p
 guard_source();D=P/'results';D.mkdir(exist_ok=False);bullet=ROOT/'build/bullet-inspect/src';header_hashes={str(p):sha(p.read_bytes())for p in bullet.rglob('*.h')};header_hashes.update({str(p):sha(p.read_bytes())for p in (ROOT/'build/spatial/_deps/json-src/single_include').rglob('*.hpp')});static=[ROOT/'build/spatial/_deps/bullet-build/src'/v for v in ['BulletDynamics/libBulletDynamics.a','BulletCollision/libBulletCollision.a','LinearMath/libLinearMath.a']];libhashes={str(p):sha(p.read_bytes())for p in static};production=ROOT/'build/spatial/spatial_runner';production_hash=sha(production.read_bytes())
 def guard():
  guard_source();assert sha(production.read_bytes())==production_hash,'Production binary changed'
  for p,h in {**header_hashes,**libhashes}.items():assert sha(Path(p).read_bytes())==h,p
 binary=D/'cache_control';cmd=['c++','-std=c++17','-O2','-Wall','-Wextra','-Wpedantic','-ffp-contract=off','-DBT_USE_DOUBLE_PRECISION','-I'+str(bullet),'-I'+str(ROOT/'build/spatial/_deps/json-src/single_include'),str(P/'control.cpp')]+[str(p)for p in static]+['-o',str(binary)]
 guard();compile_result=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);(D/'compile.json').write_text(json.dumps(dict(exit_code=compile_result.returncode,stdout=compile_result.stdout,stderr=compile_result.stderr),indent=2)+'\n');guard()
 if compile_result.returncode:raise RuntimeError('Control compilation failed; retained receipt')
 binary_hash=sha(binary.read_bytes());runtime={};ldd=subprocess.check_output(['ldd',str(binary)],text=True)
 for path in re.findall(r'^\s*\S+\s+=>\s+(/\S+)',ldd,re.M):runtime[str(Path(path).resolve())]=sha(Path(path).read_bytes())
 provenance=dict(execution_source_commit=source,source_sha256={p:sha((ROOT/p).read_bytes())for p in paths},compile_command=cmd,compiler_version=subprocess.check_output(['c++','--version'],text=True),header_hashes=header_hashes,static_library_hashes=libhashes,runtime_library_hashes=runtime,production_binary_hash=production_hash,control_binary_hash=binary_hash,scope='Read-only libraries and separate research executable; descriptive concurrent workload only.')
 (D/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 with zipfile.ZipFile(D/'source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(ROOT/p,p)
 with open(D/'stdout.jsonl','w')as log:r=subprocess.run([str(binary),str(P/'fixture.json'),str(D/'controls.json')],stdout=log,stderr=subprocess.PIPE,text=True)
 guard();assert sha(binary.read_bytes())==binary_hash
 for p,h in runtime.items():assert sha(Path(p).read_bytes())==h,p
 control=json.loads((D/'controls.json').read_text())if(D/'controls.json').exists()else{};passed=bool(r.returncode==0 and control.get('passed')and control.get('controls')==plan['expected_control_count']);receipt=dict(passed=passed,exit_code=r.returncode,stderr=r.stderr,controls=control.get('controls'),production_unchanged=True,scope='Cache-construction control only, no physical-world simulation or trajectory qualification.');(D/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
