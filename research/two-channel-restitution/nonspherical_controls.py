"""Off-centre box impacts record normal/tangent restitution and actual wrench."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from spatial_engine import run, energy
H=Path(__file__).resolve().parent;D=H/'nonspherical-controls';D.mkdir(exist_ok=False);records=[]
for angles in ((13,20,0),(-17,9,31),(7,-23,-11)):
 rotation=Rotation.from_euler('xyz',angles,degrees=True);Q=rotation.as_matrix();ext=np.array([.08,.06,.05]);height=float(np.abs(Q[2])@ext)
 body={'position':[0,0,height-1e-12],'orientation':rotation.as_quat().tolist(),'velocity':[.3,.2,-1.],'friction':2.,'shapes':[{'kind':'box','half_extents':ext.tolist(),'density':1/(8*np.prod(ext))}]}
 scene={'duration':1e-6,'gravity':[0,0,0],'bodies':[{'type':'kinematic','position':[0,0,-.1],'friction':1.,'shapes':[{'kind':'box','half_extents':[2,2,.1]}]},body]}
 result=run(scene,dt=1e-6,primary_steps=1,iterations=4096,travel_fraction=0,solver='coulomb',kinematic_contact_phase='start',position_stabilization='split_translation_combined',normal_restitution=.6,tangential_restitution=.6,record_contact_impacts=True)
 impacts=result['restitution_contact_impacts'];errors=[]
 for c in impacts:
  before=np.array(c['contact_velocity_before_normal_tangent_m_s']);after=np.array(c['contact_velocity_after_normal_tangent_m_s']);errors.append(float(np.max(abs(after+.6*before))))
  point=np.array(c['point_world_m']);impulse=np.array(c['impulse_world_kg_m_s']);n=np.array(c['normal']);normal=float(impulse@n);tangent=impulse-normal*n;assert np.linalg.norm(tangent)<=2*normal+1e-8
 # Independently derive total world momentum/angular impulse on body1.
 p=np.array(result['states']);I=np.array(result['inertia_body_kg_m2'][1]);inertia=Q@I@Q.T
 expected_v=np.array(scene['bodies'][1]['velocity'],dtype=float);angular_impulse=np.zeros(3)
 for c in impacts:
  impulse=np.array(c['impulse_world_kg_m_s'])*(1 if c['body_a']==1 else -1)
  expected_v+=impulse;angular_impulse+=np.cross(np.array(c['point_world_m'])-np.array(body['position']),impulse)
 v_error=float(np.max(abs(p[-1,1,7:10]-expected_v)));w_error=float(np.max(abs(inertia@p[-1,1,10:13]-angular_impulse)))
 change=float(energy(result)[-1]-energy(result)[0]-result['boundary_work_J'])
 record={'angles_deg':angles,'contact_count':len(impacts),'contact_restitution_error':max(errors),'linear_impulse_error':v_error,'angular_impulse_error':w_error,'kinetic_change_minus_work_J':change,'passed':bool(len(impacts)>0 and max(errors)<1e-8 and v_error<1e-8 and w_error<1e-8 and change<=1e-8)};records.append(record)
 (D/('_'.join(map(str,angles))+'.json')).write_text(json.dumps({'scene':scene,'result':result,'checks':record},indent=2)+'\n');print(record,flush=True)
summary={'passed':all(r['passed'] for r in records),'records':records,'scope':'Off-centre rotated-box endpoint restitution and full-inertia wrench checks; not measured shape-dependent material validation.'};(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');assert summary['passed']
