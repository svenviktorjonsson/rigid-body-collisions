import argparse, hashlib, json, os, platform, shutil, subprocess, time
from pathlib import Path
import numpy as np
from audit import footprint
from model import evaluate

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    here=Path(__file__).resolve().parent
    shapes={kind:footprint(kind,8,16) for kind in ['interval','hertz','ellipse','irregular']}
    patches=out/'patches.txt'
    with patches.open('w') as stream:
        stream.write(f'{len(shapes)}\n')
        for name,patch in shapes.items():
            stream.write(f'{name} {len(patch.points)}\n')
            for (x,y),w in zip(patch.points,patch.weights):stream.write(f'{x:.17g} {y:.17g} {w:.17g}\n')
    binary=out/'kernel'
    command=['g++','-std=c++17','-O3','-Wall','-Wextra','-Werror',str(here/'kernel.cpp'),'-o',str(binary)]
    compile_result=subprocess.run(command,text=True,capture_output=True)
    (out/'compile.stdout').write_text(compile_result.stdout);(out/'compile.stderr').write_text(compile_result.stderr)
    if compile_result.returncode:raise RuntimeError('Compile failed; logs retained')
    start=time.time()
    with (out/'native.json').open('w') as stream,(out/'native.stderr').open('w') as errors:
        run=subprocess.run([str(binary),str(patches)],stdout=stream,stderr=errors)
    if run.returncode:raise RuntimeError('Native run failed; raw output retained')
    result=json.loads((out/'native.json').read_text());worst=0.
    for control in result['controls']:
        r=evaluate(shapes[control['shape']],control['compression'],control['velocity'],control['omega'],10000,20,.4)
        expect=np.r_[r['force'],r['moment']]
        error=float(np.max(np.abs(expect-control['wrench'])/np.maximum(1,np.abs(expect))))
        worst=max(worst,error);assert error<1e-11
    receipt=dict(pass_=True,scope='Synthetic frozen local pressure patch including indexed body gather/scatter; no scene detection or coupled time stepping',
                 compile_command=command,compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
                 machine=platform.platform(),cpu=platform.processor(),wall_seconds=time.time()-start,
                 native_python_controls=len(result['controls']),maximum_scaled_native_python_error=worst,
                 source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [here/'kernel.cpp',here/'benchmark.py',here/'model.py',here/'audit.py']},
                 binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),patches_sha256=hashlib.sha256(patches.read_bytes()).hexdigest(),
                 timing='5 median trials; baseline then compact (order bias possible); allocations excluded; output resets, ordered gather/scatter included; same prescribed pressure sites',
                 production_adoption=False,experimental_validation=False)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
