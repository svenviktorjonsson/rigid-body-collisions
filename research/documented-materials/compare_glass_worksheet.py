"""Fixed published inputs versus author's original glass-binary worksheet.

No parameter optimization. Cached contact tangential outputs include rotation
reconstructed by the author using angular momentum, not measured spin vectors.
"""
from pathlib import Path
import json,sys,math,hashlib
import numpy as np,xlrd
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parents[1]));CACHE=Path('/home/viktor/.cache/physics-documented-materials-20261006')
from material_profiles import documented_profile,run_documented_pair
from research.spatial_scenes import sphere
from spatial_engine import energy
profile=documented_profile('glass-soda-binary');path=CACHE/'3mmglass-binary-source';s=xlrd.open_workbook(str(path)).sheet_by_index(0);D=P/'glass-worksheet-comparison';D.mkdir(exist_ok=False);records=[]
R=profile.metadata['sphere_diameter_m']/2;mass=4*math.pi*R**3*profile.metadata['sphere_density_kg_m3']/3
for i in range(s.nrows):
 if s.cell_value(i,0)!='PHOTO ID':continue
 en_obs=float(s.cell_value(i+5,8));gt=float(s.cell_value(i+5,13));gt_obs=float(s.cell_value(i+5,15));gn=float(s.cell_value(i+6,13));gn_obs=-en_obs*gn
 scene={'duration':1e-7,'gravity':[0,0,0],'bodies':[sphere([0,0,R-1e-12],radius=R,mass=mass,velocity=[gt/2,0,gn/2]),sphere([0,0,-R],radius=R,mass=mass,velocity=[-gt/2,0,-gn/2])]}
 result=run_documented_pair(scene,profile.id,dt=1e-7,primary_steps=1,iterations=4096,travel_fraction=0,kinematic_contact_phase='start',position_stabilization='split_translation_combined',record_contact_impacts=True);state=np.array(result['states'])[-1];normal_pred=float(state[0,9]-state[1,9]);tcenter_pred=float(state[0,7]-state[1,7]);gt_pred=float(tcenter_pred-R*(state[0,11]+state[1,11]));tcenter_obs=gt+(gt_obs-gt)/3.5
 # Recover COM-relative tangent by undoing the author's angular-momentum
 # reconstruction under homogeneous-sphere I; not a new inferred spin target.
 delta=float(energy(result)[-1]-energy(result)[0]-result.get('boundary_work_J',0));assert delta<=1e-10
 jt=np.clip(-(1+profile.tangential_restitution)*gt/(7/mass),-profile.sliding_friction*(1+profile.normal_restitution)*(-gn)*mass/2,profile.sliding_friction*(1+profile.normal_restitution)*(-gn)*mass/2)
 analytic_center=gt+2*jt/mass;analytic_gt=gt+7*jt/mass
 assert abs(gt_pred-analytic_gt)<1e-7 and abs(tcenter_pred-analytic_center)<1e-7 and abs(normal_pred+profile.normal_restitution*gn)<1e-7
 record={'source_excel_row':i+1,'roll':s.cell_value(i,2),'frame':s.cell_value(i,4),'incoming_relative_normal_m_s':gn,'incoming_relative_tangent_m_s':gt,'source_normal_after_m_s':gn_obs,'source_tangent_contact_after_m_s':gt_obs,'source_tangent_center_after_m_s':tcenter_obs,'native_normal_after_m_s':normal_pred,'native_tangent_contact_after_m_s':gt_pred,'native_tangent_center_after_m_s':tcenter_pred,'normal_error_m_s':normal_pred-gn_obs,'tangent_center_error_m_s':tcenter_pred-tcenter_obs,'contact_tangent_error_m_s':gt_pred-gt_obs,'energy_change_J':delta,'source_normal_ratio':en_obs,'actual_spin_independently_measured':False};records.append(record)
 (D/f"row{i+1}.json").write_text(json.dumps({'source_record':record,'scene':scene,'result':result},indent=2)+'\n')
assert len(records)==24
metric=lambda key:float(np.sqrt(np.mean([r[key]**2 for r in records])))
speednorm=np.array([math.hypot(r['incoming_relative_normal_m_s'],r['incoming_relative_tangent_m_s']) for r in records]);errs=np.array([[r['normal_error_m_s'],r['tangent_center_error_m_s']] for r in records])
summary={'profile_id':profile.id,'fixed_parameters':profile.solver_parameters,'case_count':24,'native_analytic_energy_pass_count':24,'normal_rmse_m_s':metric('normal_error_m_s'),'center_tangent_rmse_m_s':metric('tangent_center_error_m_s'),'reconstructed_contact_tangent_rmse_m_s':metric('contact_tangent_error_m_s'),'normalised_translation_joint_rmse':float(np.sqrt(np.mean((errs/speednorm[:,None])**2))),'source_url':'https://grainflowresearch.mae.cornell.edu/impact/data/Results-3mmglass-binary','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'profile_source':'https://grainflowresearch.mae.cornell.edu/impact/data/Impact%20Results.html','no_project_coefficients_fitted':True,'independent_parameter_dataset':False,'records':records,'limitations':['Published coefficient chart and worksheet may share original characterization trials; this is experimental reproduction, not independent cross-dataset validation.','Source cached quantities retain author contact-frame and angular-momentum reconstruction; spin not independently measured.','Catalog nominal diameter used, sourceindividualradii vary slightly.','No exact reproduction of photographed3Dpositions or accelerations; reduced instantaneous paired input velocities taken from worksheet.']}
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Fixed published glass coefficients:24worksheet comparisons');print('normal RMSE',summary['normal_rmse_m_s'],'center tangent RMSE',summary['center_tangent_rmse_m_s'],'normalized joint',summary['normalised_translation_joint_rmse'])
