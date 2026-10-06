"""One-step analytical friction impacts and unmodified-solver state parity."""
import json,os
from pathlib import Path
import numpy as np
from rigid_engine import run
from research.container_scenes import ball
from research.rigid_scenes import rectangle,body
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'plan.json').read_text());D=H/'controls-v2';D.mkdir(exist_ok=False)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
records=[]
for shape in ['circle','square']:
 for sign in [-1,1]:
  for wall_first in [False,True]:
   vx,vy=sign*.5,sign*.1
   wall={'type':'kinematic','position':[0,-.1],'velocity':[vx,vy],'polygons':[rectangle(2,.1,friction=.4,restitution=0)]}
   dynamic=ball((0,.11-1e-12),velocity=(1,-1),friction=.4) if shape=='circle' else body(rectangle(.2,.1,density=12.5,friction=.4,restitution=0),(0,.12-1e-12),(1,-1))
   scene={'id':f'{shape}_{sign}_{wall_first}','duration':1e-6,'gravity':[0,0],'collision_skin_m':.01,'analytic_kinematics':True,'bodies':[wall,dynamic] if wall_first else [dynamic,wall]}
   settings={'dt':1e-6,'primary_steps':1,'substeps':1,'backend':'block','position_iterations':12}
   result=run(scene,binary=ROOT/plan['binary'],**settings)
   os.environ['PHYSICS_DISABLE_JOINT']='1'
   try:disabled=run(scene,binary=ROOT/plan['binary'],**settings)
   finally:os.environ.pop('PHYSICS_DISABLE_JOINT')
   original=run(scene,binary=ROOT/'build/rigid_double_tight/rigid_runner',**settings)
   assert np.asarray(disabled['states']).tobytes()==np.asarray(original['states']).tobytes()
   assert np.asarray(disabled['kinematic_states']).tobytes()==np.asarray(original['kinematic_states']).tobytes()
   mass=result['mass'][0];I=result['inertia'][0];state=np.asarray(result['states']);first,last=state[0,0],state[-1,0]
   pn=mass*(1+vy)
   if shape=='circle':pt=-min((1-vx)/(1/mass+.1**2/I),.4*pn);expected=[1+pt/mass,vy,.1*pt/I]
   else:pt=-.4*pn;expected=[1+pt/mass,vy,0.]
   error=float(np.max(abs(last[3:6]-expected)));assert error<=1e-8,(scene['id'],last[3:6],expected,error)
   momentum=mass*(last[3:5]-first[3:5]);expected_work=float(np.dot([vx,vy],momentum));work=result['boundary_work_J'];assert abs(work-expected_work)<=1e-10
   kinetic=lambda s:.5*(mass*np.dot(s[3:5],s[3:5])+I*s[5]**2)
   passive=kinetic(last)-kinetic(first)-work;assert passive<=1e-10
   record={'case':scene['id'],'analytic_velocity_spin_error_max':error,'independent_wall_work_error_J':abs(work-expected_work),'kinetic_change_minus_wall_work_J':float(passive),'disabled_original_state_bytes_exact':True};records.append(record)
   save(D/(scene['id']+'.json'),{'scene':scene,'result':result,'disabled':disabled,'original':original,'expected_velocity_spin':expected,'checks':record});print(record,flush=True)
save(D/'summary.json',{'passed':True,'records':records,'scope':'Analytic single-point disk and two-point flat square friction impacts, signed translating walls, both body orders, original-solver bypass parity. Does not qualify global trajectories.'})
