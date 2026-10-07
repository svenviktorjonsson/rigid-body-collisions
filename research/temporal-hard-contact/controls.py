"""Independent wall impulse/work and observer-state parity controls."""
import copy,json,os
from pathlib import Path
import numpy as np
from rigid_engine import run
from research.container_scenes import ball
from research.rigid_scenes import rectangle
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'plan.json').read_text());D=H/'controls-v3';D.mkdir(exist_ok=False)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
records=[]
for omega in [0.,.5,-.5]:
 for sign in [-1,1]:
  for wall_first in [True,False]:
   velocity=[sign*.5,sign*.1] if omega==0 else [0.,0.]
   wall={'type':'kinematic','position':[0,-.1],'velocity':velocity,'omega':omega,'polygons':[rectangle(2,.1,density=1,friction=.4,restitution=0)]}
   dynamic=ball((.4,.11),velocity=(1,-1),friction=.4)
   scene={'id':f'work_{omega}_{sign}_{wall_first}','duration':.002,'gravity':[0,0],'analytic_kinematics':True,'collision_skin_m':.01,'bodies':[wall,dynamic] if wall_first else [dynamic,wall]}
   settings={'dt':.002,'primary_steps':200,'substeps':8,'backend':'temporal','binary':ROOT/plan['binary']}
   observed=run(scene,**settings);os.environ['PHYSICS_DISABLE_OBSERVER']='1'
   try:disabled=run(scene,**settings)
   finally:os.environ.pop('PHYSICS_DISABLE_OBSERVER')
   a=np.asarray(observed['states']);b=np.asarray(disabled['states']);assert a.tobytes()==b.tobytes();assert np.asarray(observed['kinematic_states']).tobytes()==np.asarray(disabled['kinematic_states']).tobytes()
   first,last=a[0,0],a[-1,0];mass=observed['mass'][0];inertia=observed['inertia'][0]
   momentum=mass*(last[3:5]-first[3:5]);r0=first[:2]-np.array(wall['position']);r1=last[:2]-np.array(wall['position'])
   angular=inertia*(last[5]-first[5])+mass*(r1[0]*last[4]-r1[1]*last[3]-r0[0]*first[4]+r0[1]*first[3])
   expected=float(np.dot(velocity,momentum)+omega*angular);work=observed['boundary_work_J'];error=abs(expected-work)
   # Angular momentum includes discrete drift; bound its independently measured defect.
   assert error<2e-5,(scene['id'],expected,work,error)
   kinetic=lambda s:.5*(mass*np.dot(s[3:5],s[3:5])+inertia*s[5]**2)
   passive=kinetic(last)-kinetic(first)-work;assert passive<=1e-8
   assert observed['boundary_impulse_points']>0 and observed['friction_impulse_abs_kg_m_s']>0
   save(D/(scene['id']+'.json'),{'scene':scene,'result':observed,'observer_disabled_result':disabled,'expected_work_from_independent_body_momentum_J':expected,'work_error_J':error,'kinetic_change_minus_boundary_work_J':passive})
   records.append({'case':scene['id'],'state_bytes_exact':True,'expected_work_J':expected,'observed_work_J':work,'work_error_J':error,'passivity_J':passive})
   print(records[-1],flush=True)
save(D/'summary.json',{'passed':True,'records':records,'scope':'Independent dynamic-body impulse/angular-momentum work identity, both body orders, signed translation and rotation, friction, observer on/off state parity. These controls do not qualify world trajectories.'})
