"""Compile only the prescribed-motion runner against the frozen planar solver."""
import hashlib,json,shutil,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
OLD=ROOT/'build/rigid_double_global_union_v1';D=ROOT/'build/rigid_double_planar_clock_v1'
D.mkdir(exist_ok=True);assert not (D/"rigid_runner").exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=json.loads((ROOT/'research/planar-global-rounded-union/build-receipt.json').read_text())
assert sha(OLD/'rigid_runner')==receipt['binary_sha256']
assert sha(OLD/'precision-source.json')==receipt['manifest_sha256']
lib=ROOT/'build/spatial/_deps/bullet-build/src'
libraries=[OLD/'box2d-2.4.1/bin/libbox2d.a',lib/'BulletDynamics/libBulletDynamics.a',lib/'BulletCollision/libBulletCollision.a',lib/'LinearMath/libLinearMath.a']
headers=list((OLD/'source/box2d-2.4.1/include').rglob('*.h'))
guards={str(p):sha(p) for p in [*libraries,*headers,OLD/'source/compat2.h',H/'runner.cpp',Path(__file__)]}
command=['c++','-std=c++17','-O3','-DNDEBUG','-Wall','-Wextra','-Wpedantic','-ffp-contract=off','-DRIGID_BLOCK_BACKEND','-DRIGID_DOUBLE_PRECISION','-I'+str(OLD/'source/box2d-2.4.1/include'),'-I'+str(OLD/'source'),str(H/'runner.cpp'),'-o',str(D/'rigid_runner'),*map(str,libraries),'-l:liblapack.so.3','-l:libblas.so.3']
r=subprocess.run(command,text=True,capture_output=True)
(H/'compile.stdout').write_text(r.stdout);(H/'compile.stderr').write_text(r.stderr)
r.check_returncode();assert all(sha(p)==digest for p,digest in guards.items())
manifest=json.loads((OLD/'precision-source.json').read_text())
manifest['reused_planar_build']=str(OLD);manifest['reused_manifest_sha256']=receipt['manifest_sha256']
manifest['analytic_clock_runner_sha256']=sha(H/'runner.cpp');manifest['runner_only_build_inputs']=guards
(D/'precision-source.json').write_text(json.dumps(manifest,indent=2)+'\n')
(H/'build-receipt.json').write_text(json.dumps({'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'binary_sha256':sha(D/'rigid_runner'),'manifest_sha256':sha(D/'precision-source.json'),'command':command,'input_guards':guards,'scope':'Runner-only exact prescribed-motion arithmetic; frozen earlier Box2D contact library unchanged.'},indent=2)+'\n')
