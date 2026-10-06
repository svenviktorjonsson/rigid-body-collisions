"""Native two-channel restitution, friction-cap, spin and backward-parity checks."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from rigid_engine import run as planar_run
from spatial_engine import run as spatial_run
from research.container_scenes import ball
from research.rigid_scenes import rectangle
from research.spatial_scenes import sphere
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'controls';D.mkdir(exist_ok=False)
save=lambda p,d:p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');records=[]
for dim in (2,3):
 for radius in (.023,.029,.05):
  for en,et in ((0.,0.),(.78,.49),(1.,1.),(.6,-1.)):
   for spin in (-2.,0.,2.):
    for mu in (.05,1.):
     mass=1.;alpha=.5 if dim==2 else .4;I=alpha*mass*radius**2;omega=spin/radius
     if dim==2:
      dyn=ball((0,radius+.01-1e-12),radius=radius,velocity=(1,-1),friction=mu);dyn['omega']=-omega
      scene={'duration':1e-6,'gravity':[0,0],'collision_skin_m':.01,'bodies':[{'type':'kinematic','position':[0,-.1],'polygons':[rectangle(2,.1,friction=mu,restitution=0)]},dyn]}
      result=planar_run(scene,dt=1e-6,primary_steps=1,substeps=1,backend='block',position_iterations=12,binary=ROOT/'build/rigid_double_restitution_v1/rigid_runner',normal_restitution=en,tangential_restitution=et)
      out=np.array(result['states'])[-1,0,3:6]
     else:
      scene={'duration':1e-6,'gravity':[0,0,0],'bodies':[{'type':'kinematic','position':[0,0,-.1],'friction':1.,'shapes':[{'kind':'box','half_extents':[2,2,.1]}]},sphere([0,0,radius-1e-12],radius=radius,mass=mass,velocity=[1,0,-1],omega=[0,omega,0],friction=mu)]}
      result=spatial_run(scene,dt=1e-6,primary_steps=1,iterations=4096,travel_fraction=0,solver='coulomb',kinematic_contact_phase='start',position_stabilization='split_translation_combined',normal_restitution=en,tangential_restitution=et)
      out=np.array(result['states'])[-1,1,[7,9,11]]
     pn=mass*(1+en);pt=float(np.clip(-(1+et)*(1-spin)/(1/mass+radius**2/I),-mu*pn,mu*pn))
     finalspin=omega-radius*pt/I
     expected=np.array([1+pt/mass,en,-finalspin if dim==2 else finalspin]);error=float(np.max(abs(out-expected)))
     before=mass+.5*I*omega**2;after=.5*mass*(out[0]**2+out[1]**2)+.5*I*out[2]**2
     record={'dimension':dim,'radius_m':radius,'normal_restitution':en,'tangential_restitution':et,'peripheral_spin_m_s':spin,'mu':mu,'max_velocity_spin_error':error,'kinetic_change_J':float(after-before),'passed':bool(error<1e-7 and after-before<1e-9)}
     records.append(record);save(D/f'{dim}_{radius}_{en}_{et}_{spin}_{mu}.json',{'scene':scene,'result':result,'expected':expected.tolist(),'checks':record})
parity=[]
for p in sorted((ROOT/'research/rubber-ball-calibration/synthetic-size-spin-controls').glob('[23]d_r*.json')):
 old=json.loads(p.read_text());dim=old['checks']['dimension'];scene=old['scene']
 if dim==2:result=planar_run(scene,dt=1e-6,primary_steps=1,substeps=1,backend='block',position_iterations=12,binary=ROOT/'build/rigid_double_restitution_v1/rigid_runner')
 else:result=spatial_run(scene,dt=1e-6,primary_steps=1,iterations=4096,travel_fraction=0,solver='coulomb',kinematic_contact_phase='start',position_stabilization='split_translation_combined')
 exact=np.array(old['result']['states']).tobytes()==np.array(result['states']).tobytes();assert exact,p
 parity.append({'case':p.name,'default_state_bytes_exact':exact})
summary={'passed':all(r['passed'] for r in records),'case_count':len(records),'records':records,'default_parity':parity,'scope':'Native two-channel restitution synthetic controls. No many-body trajectory qualification or independent measured material validation.'}
save(D/'summary.json',summary);print('two-channel cases',len(records),'passed',sum(r['passed'] for r in records),'parity',len(parity),flush=True)
assert summary['passed']
