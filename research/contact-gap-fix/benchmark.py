"""Native supported-core equivalence and indexed batch cost; no 2x gate."""
import argparse,hashlib,json,platform,subprocess
from pathlib import Path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controls',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);binary=out/'supported-benchmark'
    command=['g++','-std=c++17','-O3','-Wall','-Wextra','-Werror',str(root/'benchmark.cpp'),'-o',str(binary)]
    proc=subprocess.run(command,capture_output=True,text=True)
    (out/'compile.stdout').write_text(proc.stdout);(out/'compile.stderr').write_text(proc.stderr);proc.check_returncode()
    proc=subprocess.run([str(binary),str(Path(a.controls).resolve())],capture_output=True,text=True)
    (out/'native.stdout').write_text(proc.stdout);(out/'native.stderr').write_text(proc.stderr);proc.check_returncode()
    result=json.loads(proc.stdout)
    result.update(host=platform.platform(),compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],compile_command=command,
                  scope='Independent supported-contact updates with SoA input loads/output stores; not interacting body scenes. Allocation, collision detection, frame/quaternion evolution, changing loads, impacts and group solving excluded.',
                  baseline_speedup_claimed=False,source_sha256={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'supported_kernel.h',root/'benchmark.cpp',Path(a.controls)]})
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
