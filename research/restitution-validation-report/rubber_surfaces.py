"""All four58mm Superball surface cases from Cross2010 TableI; no endpoint fitting."""
import json,math
from pathlib import Path
import numpy as np
from spatial_engine import run
from research.spatial_scenes import sphere
H=Path(__file__).resolve().parent;D=H/'rubber-surfaces';D.mkdir(exist_ok=False)
radius,mass,speed=.029,.103,4.;rows=[]
for name,en,et,S in [('granite',.78,.49,14.9),('rubber',.78,.41,14.5),('superball_disk',.78,.57,18.2),('tennis_strings',.91,-.10,9.0)]:
 variants=[]
 for angle in (24.,25.,26.):
  for tangent in (et-.01,et,et+.01):
   a=math.radians(angle);scene={'duration':1e-6,'gravity':[0,0,0],'bodies':[{'type':'kinematic','position':[0,0,-.1],'friction':1.,'shapes':[{'kind':'box','half_extents':[2,2,.1]}]},sphere([0,0,radius-1e-12],radius=radius,mass=mass,velocity=[speed*math.sin(a),0,-speed*math.cos(a)],friction=.9)]}
   result=run(scene,dt=1e-6,primary_steps=1,iterations=4096,travel_fraction=0,solver='coulomb',kinematic_contact_phase='start',position_stabilization='split_translation_combined',normal_restitution=en,tangential_restitution=tangent,record_contact_impacts=True)
   state=result['states'][-1][1];pred=state[11]/speed;analytic=(1+tangent)*math.sin(a)/(1.4*radius);assert abs(pred-analytic)<1e-7
   variant={'angle_deg':angle,'et':tangent,'predicted_spin_factor':pred,'analytic_native_error':abs(pred-analytic)};variants.append(variant)
   (D/f'{name}_a{angle}_et{tangent:.2f}.json').write_text(json.dumps({'scene':scene,'result':result,'checks':variant},indent=2)+'\n')
 low=min(v['predicted_spin_factor'] for v in variants);high=max(v['predicted_spin_factor'] for v in variants);nominal=next(v['predicted_spin_factor'] for v in variants if v['angle_deg']==25 and abs(v['et']-et)<1e-12)
 # If the point-force model were exact, this is the inertia factor required by
 # each measured endpoint. It is a consistency diagnostic, not measured inertia.
 alpha_low=(1+et-.01)*math.sin(math.radians(24))/(radius*(S+.1))-1
 alpha_high=(1+et+.01)*math.sin(math.radians(26))/(radius*(S-.1))-1
 a=math.radians(25);normal_offset=(radius*(1+et)*math.sin(a)-1.4*radius**2*S)/((1+en)*math.cos(a))
 row={'surface':name,'normal_restitution':en,'tangential_restitution':et,'observed_spin_factor':S,'observed_spin_factor_error':.1,'predicted_central_spin_factor':nominal,'predicted_range_over_angle_and_et_uncertainty':[low,high],'spin_intervals_overlap':low<=S+.1 and high>=S-.1,'inferred_alpha_range_if_pure_point_force_model':[alpha_low,alpha_high],'effective_normal_force_offset_m_diagnostic_only':normal_offset,'friction_hypothesis':.9,'friction_independently_measured':False,'native_runs':variants};rows.append(row)
summary={'source':'https://physics.usyd.edu.au/~cross/PUBLICATIONS/48.%20EnhanceBounce.pdf','table':'I','diameter_m':.058,'mass_kg':mass,'material_identity':'same reported58mm Superball across four surfaces','case_count':4,'native_run_count':36,'spin_interval_overlap_count':sum(r['spin_intervals_overlap'] for r in rows),'rows':rows,
 'one_inertia_factor_explains_all_within_reported_angle_et_spin_uncertainty':max(r['inferred_alpha_range_if_pure_point_force_model'][0] for r in rows)<=min(r['inferred_alpha_range_if_pure_point_force_model'][1] for r in rows),
 'independent_material_validation_passed':False,'offset_is_measured_contact_point':False,
 'scope':'Native endpoint-law comparison to measured normal/tangent restitution and spin. Fixed plane/sphere inertia; force history and support compliance absent. Normal-offset diagnostic is inferred from target endpoints and cannot count as validation.'}
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Rubber native runs36; measured spin intervals overlap',summary['spin_interval_overlap_count'],'/4',flush=True)
