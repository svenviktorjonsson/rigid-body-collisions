"""Analytic prescribed poses, supported bypass parity, actual impact/reversal controls."""
import json,os,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from spatial_engine import run,BINARY,energy
from research.spatial_scenes import sphere,wall_impact,touching_container
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'controls';D.mkdir(exist_ok=False);receipt=json.loads((H/'build-receipt.json').read_text());exe=Path(receipt['binary']);assert hashlib.sha256(exe.read_bytes()).hexdigest()==receipt['binary_sha256'];save=lambda p,d:p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');records=[]
scene={'duration':.12,'gravity':[0,0,0],'bodies':[{'type':'kinematic','position':[100,0,0],'velocity':[20,0,0],'omega':[0,0,5],'shapes':[{'kind':'box','half_extents':[.05,1,1]}],'velocity_schedule':[{'time_s':.04,'velocity':[-20,0,0],'omega':[0,0,-5]},{'time_s':.08,'velocity':[20,0,0],'omega':[0,0,5]}]},sphere([0,0,0],velocity=[.1,.2,.3]) ]}
for phase in ['start','end']:
 refs=[]
 for steps in [1,2,4,16]:
  settings=dict(dt=.01,primary_steps=steps,travel_fraction=0,solver='coulomb',kinematic_contact_phase=phase,position_stabilization='split_translation_combined',iterations=4096)
  result=run(scene,binary=exe,**settings);state=np.asarray(result['states'])[:,0];t=np.asarray(result['times']);u=np.where(t<=.04,t,np.where(t<=.08,.08-t,t-.08));expected_p=np.column_stack([100+20*u,np.zeros((len(t),2))]);expected_q=Rotation.from_rotvec(np.column_stack([np.zeros((len(t),2)),5*u])).as_quat();p_error=float(np.max(abs(state[:,:3]-expected_p)));q_error=float(np.max(np.minimum(np.linalg.norm(state[:,3:7]-expected_q,axis=1),np.linalg.norm(state[:,3:7]+expected_q,axis=1))));assert p_error<=1e-13 and q_error<=1e-13;assert result['analytic_clock_policy']['enabled']
  os.environ['PHYSICS_DISABLE_ANALYTIC_CLOCK']='1'
  try:disabled=run(scene,binary=exe,**settings)
  finally:os.environ.pop('PHYSICS_DISABLE_ANALYTIC_CLOCK')
  original=run(scene,binary=BINARY,**settings);exact=np.asarray(disabled['states']).tobytes()==np.asarray(original['states']).tobytes();assert exact
  if refs:assert state.tobytes()==refs[0].tobytes()
  refs.append(state);r={'phase':phase,'primary_steps':steps,'position_error_m':p_error,'quaternion_error':q_error,'prescribed_output_bytes_exact_across_levels':True,'disabled_original_state_bytes_exact':exact};records.append(r);save(D/f'pose_{phase}_{steps}.json',{'scene':scene,'result':result,'disabled':disabled,'original':original,'checks':r})
for sign in [-1,1]:
 scene=wall_impact(speed=sign*20,restitution=0);scene['bodies'][0]['position'][0]=-sign;settings=dict(dt=.01,primary_steps=1000,travel_fraction=0,solver='coulomb',kinematic_contact_phase='start',position_stabilization='split_translation_combined',iterations=4096);result=run(scene,binary=exe,**settings);error=float(np.max(abs(np.asarray(result['states'])[-1,1,7:10]-[sign*20,0,0])));assert error<=1e-8 and abs(result['boundary_work_J']-400)<=1e-8 and energy(result)[-1]-result['boundary_work_J']<=1e-8;r={'signed_wall':sign,'analytic_velocity_error_m_s':error,'actuator_work_J':result['boundary_work_J'],'original_contact_residual_m_s':result['coulomb_residual_max_m_s']};assert r['original_contact_residual_m_s']<=1e-8;records.append(r);save(D/f'impact_{sign}.json',{'scene':scene,'result':result,'checks':r})
scene,_=touching_container(side=2,speed=100,duration=.02);scene['bodies'][0]['velocity_schedule']=[{'time_s':.01,'velocity':[-100,0,0]}];result=run(scene,binary=exe,dt=.01,primary_steps=1,travel_fraction=0,solver='normal_coupled',kinematic_contact_phase='start',position_stabilization='velocity_only');state=np.asarray(result['states']);assert np.max(abs(state[1,1:,7:10]-[100,0,0]))<=1e-8 and np.max(abs(state[2,1:,7:10]-[-100,0,0]))<=1e-8 and np.max(abs(state[2,:,:3]-state[0,:,:3]))<=1e-8;save(D/'reversal.json',{'scene':scene,'result':result,'passed':True});records.append({'touching8_reversal':True});save(D/'summary.json',{'passed':True,'records':records,'scope':'Prescribed-pose/reversal/impact controls and disabled original state parity; no world/refinement/performance acceptance.'});print('Analytic clock controls PASS',len(records),flush=True)
