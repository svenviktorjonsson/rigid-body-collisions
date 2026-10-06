"""Diagnostic torque balance for Cross2010 TableI; no fitted forward corrections."""
from pathlib import Path
import math,json
P=Path(__file__).resolve().parent;rows=[]
# Transcribed TableI values; primary source verified. Radii/masses are reported.
for ball,R,m,values in [('Superball',.029,.103,[('granite',.78,.49,14.9),('rubber',.78,.41,14.5),('Superball pad',.78,.57,18.2),('tennis strings',.91,-.10,9.0)]),('golf',.0214,.045,[('granite',.87,.03,14.8),('rubber',.59,.51,20.5),('Superball pad',.81,.55,23.6),('tennis strings',.91,-.01,15.0)])]:
 for surface,en,et,S in values:
  sin=math.sin(math.radians(25));cos=math.cos(math.radians(25));pred=(1+et)*sin/(1.4*R)
  lo=(1+et-.01)*math.sin(math.radians(24))/(1.4*R);hi=(1+et+.01)*math.sin(math.radians(26))/(1.4*R)
  # q is extra horizontal-axis moment impulse per m*v. Eliminate tangential
  # impulse using measured COM/contact restitution identity. This is an
  # inferred missing-wrench diagnostic, not a measured patch offset.
  q=1.4*R*R*S-R*(1+et)*sin
  d=-q/((1+en)*cos)
  rows.append({'ball':ball,'radius_m':R,'mass_kg':m,'surface':surface,'normal_restitution':en,'tangential_restitution':et,'observed_spin_factor_rad_m':S,'observed_error':.1,'point_law_spin_factor_rad_m':pred,'point_law_range':[lo,hi],'overlap':lo<=S+.1 and hi>=S-.1,'inferred_extra_horizontal_moment_per_mass_speed_m':q,'inferred_normal_force_offset_m':d,'offset_independently_measured':False,'homogeneous_sphere_inertia_assumed':True})
result={'source':'https://physics.usyd.edu.au/~cross/PUBLICATIONS/48.%20EnhanceBounce.pdf','table':'I','case_count':8,'rows':rows,'central_values_are_conditional_predictions':True,'no_forward_offset_correction_applied':True,'same_pair_independent_friction_validated':False,'interpretation':'Both ball types on the Superball pad underpredict spin outside propagated angle/tangent-rest uncertainty under homogeneous sphere inertia. Required additional moment is an inferred diagnostic. Golf inertia is not measured; cannot attribute discrepancy uniquely to patch deformation.'}
(P/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
for r in rows:print(r['ball'],r['surface'],'pred',round(r['point_law_spin_factor_rad_m'],3),'measured',r['observed_spin_factor_rad_m'],'overlap',r['overlap'],'d mm',round(1000*r['inferred_normal_force_offset_m'],3))
