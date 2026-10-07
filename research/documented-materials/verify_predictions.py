"""Native predictions with unchanged documented material triples; no fits."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[1]))
from material_profiles import documented_profile,run_documented_pair,MaterialDataUnavailable
from research.spatial_scenes import sphere
from spatial_engine import energy
D=P/'native-controls';D.mkdir(exist_ok=False);profiles=json.loads((P/'catalog.json').read_text())['profiles'];records=[]
# Profiles with documented density: native geometry uses published size/density.
for row in profiles:
 if row['sphere_density_kg_m3'] is None:continue
 profile=documented_profile(row['id']);R=row['sphere_diameter_m']/2;m=4*math.pi*R**3*row['sphere_density_kg_m3']/3
 for ratio in [.03,.3,1.]:
  for spin in [-.3,0,.3]:
   name=f"{profile.id}_t{ratio}_spin{spin}";vn=.2;vt=vn*ratio
   if row['geometry']=='sphere_plane':
    scene={'duration':1e-7,'gravity':[0,0,0],'bodies':[{'type':'kinematic','position':[0,0,-.01],'shapes':[{'kind':'box','half_extents':[.1,.1,.01]}]},sphere([0,0,R-1e-12],radius=R,mass=m,velocity=[vt,0,-vn],omega=[0,spin/R,0])]}
    jn=(1+profile.normal_restitution)*m*vn;u=vt-spin;jt=np.clip(-(1+profile.tangential_restitution)*u*m/3.5,-profile.sliding_friction*jn,profile.sliding_friction*jn)
    predicted=np.array([vt+jt/m,-vn+jn/m,spin/R-jt/(.4*m*R)])
   else:
    scene={'duration':1e-7,'gravity':[0,0,0],'bodies':[sphere([0,0,R-1e-12],radius=R,mass=m,velocity=[vt/2,0,-vn/2],omega=[0,spin/R,0]),sphere([0,0,-R],radius=R,mass=m,velocity=[-vt/2,0,vn/2])]}
    jn=(1+profile.normal_restitution)*vn/(2/m);u=vt-spin;jt=np.clip(-(1+profile.tangential_restitution)*u/(7/m),-profile.sliding_friction*jn,profile.sliding_friction*jn)
    predicted=np.array([vt/2+jt/m,-vn/2+jn/m,spin/R-jt/(.4*m*R)])
   result=run_documented_pair(scene,profile.id,dt=1e-7,primary_steps=1,iterations=4096,travel_fraction=0,kinematic_contact_phase='start',position_stabilization='split_translation_combined',record_contact_impacts=True)
   index=1 if row['geometry']=='sphere_plane' else 0;last=np.array(result['states'])[-1,index];actual=last[[7,9,11]];err=float(np.max(abs(actual-predicted)));delta=float(energy(result)[-1]-energy(result)[0]-result.get('boundary_work_J',0));passed=err<1e-7 and delta<=1e-10
   assert passed,(name,actual,predicted,err,delta)
   check={'case':name,'profile':profile.id,'prediction_max_error':err,'kinetic_change_minus_work_J':delta,'passed':bool(passed),'measured_trial_replay':False};records.append(check);(D/(name+'.json')).write_text(json.dumps({'scene':scene,'parameters_from_document':profile.solver_parameters,'result':result,'checks':check},indent=2)+'\n')
for missing in ['limestone-concrete','Superball-granite','rubber','steel']:
 try:documented_profile(missing)
 except MaterialDataUnavailable:pass
 else:raise AssertionError('missing/generic profile silently substituted')
summary={'case_count':len(records),'pass_count':sum(r['passed'] for r in records),'records':records,'coefficient_values_modified':False,'parameter_source':'catalog.json; published nominal values','catalog_sha256':hashlib.sha256((P/'catalog.json').read_bytes()).hexdigest(),'independent_experimental_motion_validated':False,'interpretation':'Integration/analytic/energy controls with documented coefficients, sizes and densities. They are not measured trial comparisons; source validity ranges are not established for every spin/speed.'}
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Documented input native controls',len(records),'PASS',flush=True)
