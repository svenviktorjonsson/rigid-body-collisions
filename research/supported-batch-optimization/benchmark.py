"""Alternating-order paired kernel benchmark; immutable new evidence paths."""
import argparse,hashlib,json,platform,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controls',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);binary=out/'benchmark'
    sources=[ROOT/'research/supported-batch-optimization/benchmark.cpp',ROOT/'supported_backend/batch.cpp',ROOT/'supported_backend/kernel.h',ROOT/'supported_backend/batch.h',ROOT/'research/contact-gap-fix/supported_kernel.h']
    command=['g++','-std=c++17','-O3','-Wall','-Wextra','-Werror','-fopenmp',str(sources[0]),str(sources[1]),'-o',str(binary)]
    result=subprocess.run(command,capture_output=True,text=True);(out/'compile.txt').write_text(result.stdout+result.stderr);result.check_returncode()
    result=subprocess.run([str(binary),str(Path(a.controls).resolve())],capture_output=True,text=True);(out/'native.stdout').write_text(result.stdout);(out/'native.stderr').write_text(result.stderr);result.check_returncode()
    data=json.loads(result.stdout);data.update(compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],host=platform.platform(),cpu=subprocess.check_output(['lscpu'],text=True),compile_command=command,
        scope='Same physical response channels and energy/branch gates. Field loads and all 11 stores included. Preparation/validation/allocation, collision detection, moving frames, impacts, groups and contact-history update excluded.',
        target_ms=20.,target_responses=1000000,preferred_threads=4,
        source_sha256={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},controls_sha256=hashlib.sha256(Path(a.controls).read_bytes()).hexdigest())
    (out/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    for row in data['batches']:
        if row['responses']==1000000:print(row)
