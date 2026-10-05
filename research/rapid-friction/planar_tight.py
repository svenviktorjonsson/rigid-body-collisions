"""Build a separate Float64 penetration-slop diagnostic and qualify it."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    plan=json.loads((HERE/'planar-tight-plan.json').read_text());out=HERE/'results-planar-tight';out.mkdir(exist_ok=False)
    build=Path('/home/viktor/.cache/physics-planar-tight-01');build.mkdir(exist_ok=False)
    baseline=ROOT/'build/rigid_double_friction'
    source=build/'source';shutil.copytree(baseline/'source',source)
    header=source/'box2d-2.4.1/include/box2d/b2_common.h';before=header.read_bytes()
    text=header.read_text();assert text.count('(0.005 * b2_lengthUnitsPerMeter)')==1
    header.write_text(text.replace('(0.005 * b2_lengthUnitsPerMeter)','(0.000001 * b2_lengthUnitsPerMeter)'))
    provenance=json.loads((baseline/'precision-source.json').read_text());provenance['numerical_change']={'header':'box2d-2.4.1/include/box2d/b2_common.h','before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':digest(header),'linear_slop_m':plan['linear_slop_m'],'baseline_precision_manifest_sha256':digest(baseline/'precision-source.json')}
    provenance['transformed']={str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file() and p.suffix in ('.h','.cpp')}
    (build/'precision-source.json').write_text(json.dumps(provenance,indent=2)+'\n')
    commands=[['cmake','-S',str(source),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release'],['cmake','--build',str(build),'--parallel','2']]
    for i,command in enumerate(commands):
        result=subprocess.run(command,capture_output=True,text=True);(out/f'build_{i}.json').write_text(json.dumps({'command':command,'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr},indent=2)+'\n')
        if result.returncode:raise SystemExit(result.returncode)
    (out/'precision-source.json').write_bytes((build/'precision-source.json').read_bytes())
    (out/'scenes.json').write_bytes((HERE/'results-planar-shake/scenes.json').read_bytes())
    text=(HERE/'planar_resolution.py').read_text()
    text=text.replace("HERE=Path(__file__).resolve().parent","HERE=Path("+repr(str(HERE))+")")
    text=text.replace('planar-resolution-plan.json','planar-tight-plan.json')
    text=text.replace("HERE/'results/scenes.json'","HERE/'results-planar-tight/scenes.json'")
    text=text.replace("out=HERE/'results-planar-resolution';out.mkdir(exist_ok=False)","out=HERE/'results-planar-tight'")
    text=text.replace("base.ROOT/'build/rigid_double_ledger/rigid_runner'","Path("+repr(str(build/'rigid_runner'))+")")
    text=text.replace("HERE/'planar_resolution.py'","Path(__file__).resolve()")
    text=text.replace("qualified=all(e['passed'] for e in edges)","qualified=all(e['passed'] for e in edges[-2:])")
    text=text.replace("'setting':settings[0]","'setting':settings[-3]").replace("refs[0]['result']","refs[-3]['result']")
    text=text.replace("base.atomic(dest,record);return record","\n        if record['complete']:\n            record['physical']['passed'] &= record['result']['friction_impulse_abs_kg_m_s']>0\n        base.atomic(dest,record);return record")
    adapter=out/'adapter.py';adapter.write_text(text)
    raise SystemExit(subprocess.run([sys.executable,str(adapter)],cwd=ROOT).returncode)

if __name__=='__main__':main()
