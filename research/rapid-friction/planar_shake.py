"""Declare planar counterparts then use the frozen qualification/timing driver."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from research.container_scenes import container,ball

def main():
    plan=json.loads((HERE/'planar-shake-plan.json').read_text())
    destination=HERE/'results-planar-shake';destination.mkdir(exist_ok=False)
    entries={}
    for side in plan['sides']:
        half=side*.205/2+.045
        wall=container(half,half,velocity=(20,0),friction=.4)
        wall['velocity_schedule']=[{'time_s':.04,'velocity':[-20,0]},{'time_s':.08,'velocity':[20,0]}]
        disks=[ball(((i%side-(side-1)/2)*.205,(i//side-(side-1)/2)*.205),friction=.4) for i in range(side*side)]
        name=f'2d_shake{side*side}'
        entries[name]={'dimension':2,'scene':{'id':name,'duration':.12,'gravity':[0,-9.81],'collision_skin_m':.01,'analytic_kinematics':True,'container_half_extents_m':[half,half],'bodies':[wall,*disks]}}
    (destination/'scenes.json').write_text(json.dumps(entries,indent=2)+'\n')
    # Keep the independently published original iteration driver intact. This
    # generated adapter changes only declared plan/scenes/destination/binary.
    text=(HERE/'planar_resolution.py').read_text()
    text=text.replace("HERE=Path(__file__).resolve().parent","HERE=Path("+repr(str(HERE))+")")
    text=text.replace("planar-resolution-plan.json","planar-shake-plan.json")
    text=text.replace("HERE/'results/scenes.json'","HERE/'results-planar-shake/scenes.json'")
    text=text.replace("out=HERE/'results-planar-resolution';out.mkdir(exist_ok=False)","out=HERE/'results-planar-shake'")
    text=text.replace('build/rigid_double_ledger/rigid_runner','build/rigid_double_friction/rigid_runner')
    text=text.replace("'physical':base.physical(entry,result,gates)","'physical':base.physical(entry,result,gates)")
    text=text.replace("base.atomic(dest,record);return record","\n        if record['complete']:\n            record['physical']['passed'] &= record['result']['friction_impulse_abs_kg_m_s']>0\n        base.atomic(dest,record);return record")
    # Guard the generated adapter too; it is retained as actual execution source.
    text=text.replace("HERE/'planar_resolution.py'","Path(__file__).resolve()")
    adapter=destination/'adapter.py';adapter.write_text(text)
    raise SystemExit(subprocess.run([sys.executable,str(adapter)],cwd=ROOT).returncode)

if __name__=='__main__':main()
