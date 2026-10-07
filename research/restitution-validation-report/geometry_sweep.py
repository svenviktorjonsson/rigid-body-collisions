"""Fixed e_n/e_t across varied non-spherical shape, contact point and direction."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from spatial_engine import run as run3, energy, moments
from rigid_engine import run as run2
from research.rigid_scenes import rectangle,body
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'geometry-sweep';D.mkdir(exist_ok=False);records=[]
en=et=.6
for dim in (2,3):
 for shape in ('box','hull'):
  for angle in (13.,-17.,29.,-31.):
   for v in ((.3,-1.),(1.5,-.4),(-1.,-1.2)):
    for mu in (.2,2.):
     case=f'{dim}_{shape}_{angle}_{v[0]}_{mu}'
     try:
      if dim==2:
       vertices=np.array(rectangle(.08,.05)['vertices'] if shape=='box' else [[.08*math.cos(2*math.pi*k/8),.05*math.sin(2*math.pi*k/8)] for k in range(8)])
       a=math.radians(angle);Q=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]]);world=vertices@Q.T;corner=world[np.argmin(world[:,1])];height=-corner[1]
       area=.5*abs(np.sum(vertices[:,0]*np.roll(vertices[:,1],-1)-vertices[:,1]*np.roll(vertices[:,0],-1)))
       dynamic=body({'vertices':vertices.tolist(),'density':1/area,'friction':mu},(0,height+.02-1e-12),v,angle=a)
       scene={'duration':1e-6,'gravity':[0,0],'collision_skin_m':.01,'bodies':[{'type':'kinematic','position':[0,-.1],'polygons':[rectangle(2,.1,friction=mu)]},dynamic]}
       result=run2(scene,dt=1e-6,primary_steps=1,substeps=1,backend='block',position_iterations=12,normal_restitution=en,tangential_restitution=et)
       states=np.array(result['states']);last=states[-1,0];r=np.array([corner[0],corner[1]-.01]);mass=result['mass'][0];I=result['inertia'][0];impulse=mass*(last[3:5]-np.array(v));momentum_error=float(abs(I*last[5]-(r[0]*impulse[1]-r[1]*impulse[0])))
       before=np.array([v[1],v[0]]);after=np.array([last[4]+last[5]*r[0],last[3]-last[5]*r[1]]);cap=mu*impulse[1];friction_bound=abs(impulse[0])<=cap+1e-8;energy_change=float(.5*mass*np.dot(last[3:5],last[3:5])+.5*I*last[5]**2-.5*mass*np.dot(v,v));normal_error=abs(after[0]+en*before[0]);tangent_error=abs(after[1]+et*before[1]);capacity_limited=abs(abs(impulse[0])-cap)<1e-8
       point=[float(corner[0]),.01];angular_speed=abs(last[5]);contact_count=1
      else:
       angles=(angle,20+angle/4,11);rotation=Rotation.from_euler('xyz',angles,degrees=True);Q=rotation.as_matrix()
       if shape=='box':geom={'kind':'box','half_extents':[.08,.06,.05],'density':1/(8*.08*.06*.05)};verts=np.array([[x,y,z] for x in (-.08,.08) for y in (-.06,.06) for z in (-.05,.05)])
       else:
        verts=np.array([[-.08,-.05,-.045],[.09,-.04,-.035],[.06,.07,-.02],[-.05,.045,-.055],[-.035,-.02,.07],[.04,.035,.06]])
        vol,center,_,_=moments({'kind':'hull','vertices':verts.tolist()});geom={'kind':'hull','vertices':verts.tolist(),'density':1/vol};verts=verts-center
       height=-float(np.min((verts@Q.T)[:,2]));position=np.array([0,0,height-1e-12]);dynamic={'position':position.tolist(),'orientation':rotation.as_quat().tolist(),'velocity':[v[0],.2,v[1]],'friction':mu,'shapes':[geom]}
       scene={'duration':1e-6,'gravity':[0,0,0],'bodies':[{'type':'kinematic','position':[0,0,-.1],'friction':1.,'shapes':[{'kind':'box','half_extents':[2,2,.1]}]},dynamic]}
       result=run3(scene,dt=1e-6,primary_steps=1,iterations=4096,travel_fraction=0,solver='coulomb',kinematic_contact_phase='start',position_stabilization='split_translation_combined',normal_restitution=en,tangential_restitution=et,record_contact_impacts=True)
       contacts=result['restitution_contact_impacts'];normal_error=tangent_error=0.;friction_bound=True;capacity_limited=False;torque=np.zeros(3)
       for c in contacts:
        pre=np.array(c['contact_velocity_before_normal_tangent_m_s']);post=np.array(c['contact_velocity_after_normal_tangent_m_s']);normal_error=max(normal_error,float(abs(post[0]+en*pre[0])));tangent_error=max(tangent_error,float(np.linalg.norm(post[1:]+et*pre[1:])))
        impulse=np.array(c['impulse_world_kg_m_s']);n=np.array(c['normal']);pn=float(impulse@n);pt=np.linalg.norm(impulse-pn*n);friction_bound &= bool(pt<=mu*pn+1e-8);capacity_limited |= abs(pt-mu*pn)<1e-8
        torque+=np.cross(np.array(c['point_world_m'])-position,impulse*(1 if c['body_a']==1 else -1))
       state=np.array(result['states'])[-1,1];I=Q@np.array(result['inertia_body_kg_m2'][1])@Q.T;momentum_error=float(np.max(abs(I@state[10:13]-torque)));energy_change=float(energy(result)[-1]-energy(result)[0]-result['boundary_work_J']);point=contacts[0]['point_world_m'];angular_speed=float(np.linalg.norm(state[10:13]));contact_count=len(contacts)
      passed=bool(normal_error<1e-7 and (tangent_error<1e-7 or capacity_limited) and friction_bound and energy_change<=1e-8 and momentum_error<1e-7)
      check={'case':case,'dimension':dim,'shape':shape,'angle_deg':angle,'velocity':v,'mu':mu,'normal_restitution':en,'tangential_restitution':et,'contact_count':contact_count,'contact_point_world_m':point,'normal_target_error':float(normal_error),'tangent_target_error':float(tangent_error),'capacity_limited':bool(capacity_limited),'friction_bound_passed':bool(friction_bound),'angular_impulse_error':momentum_error,'kinetic_change_minus_work_J':energy_change,'outgoing_angular_speed_rad_s':float(angular_speed),'passed':passed};entry={'scene':scene,'result':result,'checks':check}
     except Exception as e:check={'case':case,'dimension':dim,'shape':shape,'passed':False,'error':repr(e)};entry={'checks':check}
     records.append(check);(D/(case+'.json')).write_text(json.dumps(entry,indent=2,allow_nan=False)+'\n')
summary={'case_count':len(records),'pass_count':sum(r['passed'] for r in records),'records':records,'coefficients_fixed':True,'fitted_material_location_or_direction_dependence':False,'actual_rock_shape_validation':False}
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Geometry fixed-coefficient checks',summary['pass_count'],'/',len(records),flush=True)
