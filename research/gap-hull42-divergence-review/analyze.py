"""Independent completed seed42 trajectory diagnostics, never native execution."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).parent;plan=json.loads((P/'plan.json').read_text());sha=lambda raw:hashlib.sha256(raw).hexdigest();assert not (P/'analysis.json').exists()
for p,expected in plan['inputs_sha256'].items():assert sha((ROOT/p).read_bytes())==expected,p
study=json.loads((ROOT/'research/hull-gap-completion/plan.json').read_text());scene=json.loads((ROOT/'research/hull-gap-completion/results/scenes.json').read_text())[plan['scene']]['scene'];runs=[json.loads((ROOT/'research/hull-gap-completion/results/checkpoints'/plan['scene']/f'reference_{i}.json').read_text())for i in range(3)];states=[np.array(r['states'])for r in runs];times=np.array(runs[0]['times']);assert all(np.array_equal(times,r['times'])for r in runs);ids=np.flatnonzero(np.array(runs[0]['mass'])>0);mass=np.array(runs[0]['mass'])[ids];inertia=np.array(runs[0]['inertia_body_kg_m2'])[ids]
for r in runs:assert r['mass']==runs[0]['mass'] and r['inertia_body_kg_m2']==runs[0]['inertia_body_kg_m2'] and r['physical_setup_id']==runs[0]['physical_setup_id'];assert r['attempt_status']=='history_complete'
provenance=json.loads((ROOT/'research/hull-gap-completion/results/checkpoints/provenance.json').read_text());assert provenance['execution_source_commit']==plan['execution_source']
with zipfile.ZipFile(ROOT/'research/hull-gap-completion/results/execution-source.zip')as z:
 for p,expected in provenance['source_hashes'].items():assert sha(z.read(p))==expected and z.read(p)==subprocess.check_output(['git','show',plan['execution_source']+':'+p],cwd=ROOT)
 source={p:z.read(p).decode()for p in ['spatial_backend/runner.cpp','spatial_engine.py','spatial_backend/shared_contact.h','spatial_backend/coulomb.h']}
limits={k:v/4 for k,v in study['trajectory_budget'].items()}
def angle(a,b):
 a=a/np.linalg.norm(a,axis=-1,keepdims=True);b=b/np.linalg.norm(b,axis=-1,keepdims=True);v=a[...,3,None]*b[...,:3]-b[...,3,None]*a[...,:3]-np.cross(a[...,:3],b[...,:3]);w=a[...,3]*b[...,3]+np.sum(a[...,:3]*b[...,:3],axis=-1);return 2*np.arctan2(np.linalg.norm(v,axis=-1),abs(w))
def rotation(q):
 q=q/np.linalg.norm(q,axis=-1,keepdims=True);x,y,z,w=[q[...,i]for i in range(4)];return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(q.shape[:-1]+(3,3))
def first(values,threshold):
 i=np.flatnonzero(np.array(values)>threshold);return dict(index=int(i[0]),time_s=float(times[i[0]]),value=float(values[i[0]]))if len(i)else None
pairs=[]
for left,right in [(0,1),(1,2),(0,2)]:
 a,b=[states[i][:,ids]for i in [left,right]];delta=dict(position_m=np.linalg.norm(a[...,:3]-b[...,:3],axis=-1),velocity_m_s=np.linalg.norm(a[...,7:10]-b[...,7:10],axis=-1),omega_rad_s=np.linalg.norm(a[...,10:13]-b[...,10:13],axis=-1),orientation_rad=angle(a[...,3:7],b[...,3:7]));perframe={k:np.sqrt(np.mean(v*v,axis=1))for k,v in delta.items()};frames=[]
 for t,time in enumerate(times):frames.append(dict(time_s=float(time),**{k:float(v[t])for k,v in perframe.items()}))
 bodies=[]
 for j,ident in enumerate(ids):bodies.append(dict(body_id=int(ident),global_RMS={k:float(np.sqrt(np.mean(v[:,j]**2)))for k,v in delta.items()},first_crossing={k:first(v[:,j],limits[k])for k,v in delta.items()},final_error={k:float(v[-1,j])for k,v in delta.items()}))
 kinA,kinB=[states[i][:,0]for i in [left,right]]
 pairs.append(dict(left=f'reference_{left}',right=f'reference_{right}',global_RMS={k:float(np.sqrt(np.mean(v*v)))for k,v in delta.items()},perframe_RMS=frames,first_quarter_budget_crossing={k:first(v,limits[k])for k,v in perframe.items()},first_measurable_difference={k:first(v,1e-10)for k,v in perframe.items()},per_body=bodies,kinematic_max_error=dict(position_m=float(np.max(np.linalg.norm(kinA[:,:3]-kinB[:,:3],axis=1))),velocity_m_s=float(np.max(np.linalg.norm(kinA[:,7:10]-kinB[:,7:10],axis=1))),omega_rad_s=float(np.max(np.linalg.norm(kinA[:,10:13]-kinB[:,10:13],axis=1))),orientation_rad=float(np.max(angle(kinA[:,3:7],kinB[:,3:7]))))))
# Actual boundary schedules are piecewise constant, including exact output-end
# positions. At reversal output endpoints the state reports the just-finished
# interval's velocity; the next update installs the schedule's new velocity.
commands=scene['bodies'][0]['velocity_schedule'];expected_position=[];expected_left_v=[]
for t in times:
 x=np.array(scene['bodies'][0]['position'],float);v=np.array(scene['bodies'][0]['velocity'],float);start=0.
 for c in commands:
  end=min(t,c['time_s']);x+=v*max(0,end-start)
  if t<=c['time_s']:break
  start=c['time_s'];v=np.array(c['velocity'],float)
 else:x+=v*max(0,t-start)
 expected_position.append(x);expected_left_v.append(v)
expected_position=np.array(expected_position);expected_left_v=np.array(expected_left_v);physical=[]
for i,(r,s)in enumerate(zip(runs,states)):
 R=rotation(s[:,ids,3:7]);v=s[:,ids,7:10];w=s[:,ids,10:13];local=np.einsum('tbij,tbj->tbi',R.transpose(0,1,3,2),w);kinetic=.5*np.sum(mass[None,:]*np.sum(v*v,axis=-1),axis=1)+.5*np.einsum('tbi,bij,tbj->t',local,inertia,local);potential=-np.einsum('tbi,i,b->t',s[:,ids,:3],scene['gravity'],mass);momentum=np.einsum('tbi,b->ti',v,mass);spin=np.einsum('tbij,bjk,tbk->tbi',R,inertia,local);L=np.sum(np.cross(s[:,ids,:3],mass[None,:,None]*v)+spin,axis=1)
 physical.append(dict(lane=f'reference_{i}',frames=[dict(time_s=float(t),kinetic_J=float(K),potential_J=float(U),linear_momentum_kg_m_s=p.tolist(),angular_momentum_kg_m2_s=l.tolist())for t,K,U,p,l in zip(times,kinetic,potential,momentum,L)],final_energy_minus_boundary_work_J=float(kinetic[-1]+potential[-1]-kinetic[0]-potential[0]-r['boundary_work_J']),boundary_work_J=r['boundary_work_J'],residual_max_m_s=r['coulomb_residual_max_m_s'],updates=r['updates'],mean_internal_h_per_frame_s=(np.diff(times)/r['updates']).tolist(),container_expected_position_max_error_m=float(np.max(np.linalg.norm(s[:,0,:3]-expected_position,axis=1))),container_expected_left_velocity_max_error_m_s=float(np.max(np.linalg.norm(s[:,0,7:10]-expected_left_v,axis=1))),pose_ledger={k:v for k,v in r.items()if k.startswith('translation_pose_')},solver_counters={k:v for k,v in r.items()if k.startswith('coulomb_')and any(x in k for x in ['solves','restarts','svd_calls','sweeps_total'])}))
out=dict(schema=plan['schema'],execution_source=plan['execution_source'],source_hash=sha(Path(__file__).read_bytes()),input_hashes=plan['inputs_sha256'],times_s=times.tolist(),quarter_budget=limits,schedule=commands,pairs=pairs,physical_lanes=physical,source_observations=dict(h_clips_schedule_events='h=std::min(h,event-t)'in source['spatial_backend/runner.cpp'],sample_ends_clip_time='std::min(left,dt/primary)'in source['spatial_backend/runner.cpp'],start_phase_updates_kinematic_velocity_before_step='b.rb->setLinearVelocity(b.v)'in source['spatial_backend/runner.cpp'],contact_solver_residual_and_trajectory_are_separate=True),limitations=['Output states every.01s bracket divergence; they cannot reveal the exact first divergent internal contact.','Counts reveal average internal h only, not exact solver/substep endpoints or contact ordering.','No same-timestep repeatability or fixed-grid comparison was executed; chaos and cache sensitivity are unproven.','All-row roots can be passive yet rigid multi-impact/friction solutions may be nonunique; no multiplicity certificate is inferred from trajectory differences.'])
(P/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(pairs=[{k:p[k]for k in ['left','right','global_RMS','first_quarter_budget_crossing','first_measurable_difference','kinematic_max_error']}for p in pairs],physical=[{k:p[k]for k in ['lane','container_expected_position_max_error_m','container_expected_left_velocity_max_error_m_s','final_energy_minus_boundary_work_J']}for p in physical]),indent=2))
