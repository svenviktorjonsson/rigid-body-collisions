"""Common endpoint pose bytes, authored schedule and disabled old arithmetic."""
import json,os
from pathlib import Path
import numpy as np
from rigid_engine import run
from research.container_scenes import ball
from research.rigid_scenes import rectangle
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'plan.json').read_text())
D=H/'pose-controls';D.mkdir(exist_ok=False);records=[];reference=None
wall={'type':'kinematic','position':[0,-.1],'velocity':[20,0],'omega':5,'polygons':[rectangle(2,.1,friction=.4,restitution=0)],'velocity_schedule':[{'time_s':.04,'velocity':[-20,0],'omega':-5},{'time_s':.08,'velocity':[20,0],'omega':5}]}
scene={'id':'analytic_clock_pose','duration':.12,'gravity':[0,0],'collision_skin_m':.01,'analytic_kinematics':True,'bodies':[wall,ball((10,10),friction=.4)]}
for primary in [1,2,4,16]:
 setting={'dt':.01,'primary_steps':primary,'substeps':128,'backend':'block','position_iterations':12}
 result=run(scene,binary=ROOT/plan['binary'],**setting)
 os.environ['PHYSICS_DISABLE_ANALYTIC_CLOCK']='1'
 try:disabled=run(scene,binary=ROOT/plan['binary'],**setting)
 finally:os.environ.pop('PHYSICS_DISABLE_ANALYTIC_CLOCK')
 old=run(scene,binary=ROOT/'build/rigid_double_global_union_v1/rigid_runner',**setting)
 assert np.asarray(disabled['states']).tobytes()==np.asarray(old['states']).tobytes()
 assert np.asarray(disabled['kinematic_states']).tobytes()==np.asarray(old['kinematic_states']).tobytes()
 states=np.asarray(result['kinematic_states'])[:,0];times=np.asarray(result['times'])
 x=np.where(times<=.04,20*times,np.where(times<=.08,.8-20*(times-.04),20*(times-.08)))
 error=float(max(np.max(abs(states[:,0]-x)),np.max(abs(states[:,1]+.1)),np.max(abs(states[:,2]-.25*x))))
 assert error<=1e-14
 raw=np.asarray(result['kinematic_states']).tobytes()
 if reference is None:reference=raw
 assert raw==reference
 record={'primary_steps':primary,'common_kinematic_endpoint_bytes_exact':True,'authored_pose_error_max':error,'disabled_old_state_bytes_exact':True};records.append(record)
 (D/f'{primary}.json').write_text(json.dumps({'result':result,'disabled':disabled,'old':old,'checks':record},indent=2)+'\n')
(D/'summary.json').write_text(json.dumps({'passed':True,'scene':scene,'records':records,'scope':'Four no-contact prescribed-motion ladders with both reversals, common endpoint byte equality and old arithmetic parity. Full trajectory accuracy still required.'},indent=2)+'\n')
