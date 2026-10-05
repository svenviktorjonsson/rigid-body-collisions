"""Independent archive and trajectory-gate audit; no production metrics imports."""
import itertools
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'research/shared-hull-followup'
SOURCE='38f407d12208654075c07315e96b9fc213612b91'
def sha(data):return hashlib.sha256(data).hexdigest()

def rotation(q):
 q=np.asarray(q);q=q/np.linalg.norm(q);v=q[:3];w=q[3]
 K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
 return (w*w-v@v)*np.eye(3)+2*np.outer(v,v)+2*w*K

def shape_integrals(shape):
 kind=shape['kind']
 if kind=='sphere':
  r=shape['radius'];V=4*np.pi*r**3/3;return V,np.zeros(3),2*V*r*r/5*np.eye(3)
 if kind=='box':
  h=np.asarray(shape['half_extents']);V=8*np.prod(h);return V,np.zeros(3),V/3*np.diag(np.sum(h*h)-h*h)
 P=np.asarray(shape['vertices']);hull=ConvexHull(P);V=0.;first=np.zeros(3);second=np.zeros((3,3))
 for face,equation in zip(hull.simplices,hull.equations):
  a,b,c=P[face]
  if np.dot(np.cross(b-a,c-a),equation[:3])<0:b,c=c,b
  volume=np.dot(a,np.cross(b,c))/6;vertices=np.asarray([a,b,c]);total=vertices.sum(axis=0)
  V+=volume;first+=volume*total/4;second+=volume*(np.outer(total,total)+vertices.T@vertices)/20
 center=first/V;second-=V*np.outer(center,center)
 return V,center,np.trace(second)*np.eye(3)-second

def body_integrals(body):
 parts=[]
 for s in body['shapes']:
  V,c,I=shape_integrals(s);rho=s.get('density',1.);R=rotation(s.get('orientation',[0,0,0,1]));c=R@c+np.asarray(s.get('center',[0,0,0]));parts.append((V*rho,c,rho*R@I@R.T))
 mass=sum(p[0] for p in parts);center=sum(m*c for m,c,I in parts)/mass;inertia=np.zeros((3,3))
 for m,c,I in parts:
  d=c-center;inertia+=I+m*((d@d)*np.eye(3)-np.outer(d,d))
 return mass,center,inertia

def trajectory_metrics(scene,result):
 states=np.asarray(result['states']);assert states.ndim==3 and states.shape[2]==13 and np.isfinite(states).all()
 g=np.asarray(scene.get('gravity',[0,0,-9.81]));E=np.zeros(len(states));sampled=-np.inf;half=scene['container_interior_half_extents_m'][0]
 for i,body in enumerate(scene['bodies']):
  geometric_mass,center,I=body_integrals(body);m=geometric_mass if body.get('type','dynamic')=='dynamic' else 0.
  assert np.isclose(result['mass'][i],m,rtol=1e-12,atol=1e-12)
  assert np.allclose(result['inertia_body_kg_m2'][i],I,rtol=1e-11,atol=1e-11)
  for j,frame in enumerate(states):
   x=frame[i];R=rotation(x[3:7]);spin=R.T@x[10:13]
   if m:E[j]+=.5*m*(x[7:10]@x[7:10])+.5*spin@I@spin-m*g@x[:3]
   if i==0:continue
   Rc=rotation(frame[0,3:7]);relative=Rc.T@(x[:3]-frame[0,:3])
   for s in body['shapes']:
    Rs=rotation(s.get('orientation',[0,0,0,1]));offset=np.asarray(s.get('center',[0,0,0]))-center
    if s['kind']=='sphere':excess=np.max(abs(relative+Rc.T@R@offset))+s['radius']-half
    else:
     vertices=np.asarray(s['vertices']) if s['kind']=='hull' else np.asarray(list(itertools.product([-1,1],repeat=3)))*s['half_extents']
     points=(vertices@Rs.T+offset)@R.T@Rc+relative
     excess=np.max(abs(points))-half+(scene.get('margin_m',0) if s['kind']=='hull' else 0)
    sampled=max(sampled,float(excess))
 return dict(quaternion_norm_error=float(np.max(abs(np.linalg.norm(states[:,:,3:7],axis=2)-1))),
             energy_change_minus_boundary_work_J=float(E[-1]-E[0]-result['boundary_work_J']),
             container_surface_excess_m=max(sampled,result['max_container_surface_excess_m']),
             sampled_container_surface_excess_m=sampled)

def trajectory_error(left,right):
 assert left['physical_setup_id']==right['physical_setup_id'] and left['mass']==right['mass'] and left['inertia_body_kg_m2']==right['inertia_body_kg_m2']
 assert np.allclose(left['times'],right['times'],atol=1e-12,rtol=0)
 ids=np.flatnonzero(np.asarray(left['mass'])>0);a=np.asarray(left['states'])[:,ids];b=np.asarray(right['states'])[:,ids]
 rms=lambda x:float(np.sqrt(np.mean(np.sum(x*x,axis=-1))))
 qa=a[:,:,3:7];qb=b[:,:,3:7];qa=qa/np.linalg.norm(qa,axis=-1,keepdims=True);qb=qb/np.linalg.norm(qb,axis=-1,keepdims=True)
 vec=qa[:,:,3,None]*qb[:,:,:3]-qb[:,:,3,None]*qa[:,:,:3]-np.cross(qa[:,:,:3],qb[:,:,:3]);scalar=np.abs(np.sum(qa*qb,axis=-1));angle=2*np.arctan2(np.linalg.norm(vec,axis=-1),scalar)
 return dict(position_m=rms(a[:,:,:3]-b[:,:,:3]),velocity_m_s=rms(a[:,:,7:10]-b[:,:,7:10]),omega_rad_s=rms(a[:,:,10:13]-b[:,:,10:13]),orientation_rad=float(np.sqrt(np.mean(angle*angle))))

def audit(study=DIRECTORY,source=SOURCE):
 study=Path(study);directory=study/'results';summary=json.loads((directory/'summary.json').read_text());plan=json.loads((study/'plan.json').read_text())
 assert summary['execution_source_commit']==source and summary['plan_sha256']==sha((study/'plan.json').read_bytes())
 assert summary['attempt_count']==summary['planned_attempt_count']==6 and summary['complete']
 for name,digest in summary['hashes'].items():assert sha((directory/name).read_bytes())==digest
 with zipfile.ZipFile(directory/'execution-source.zip') as archive:
  assert set(archive.namelist())==set(summary['source_hashes'])
  for name,digest in summary['source_hashes'].items():
   data=archive.read(name);assert sha(data)==digest
   assert data==subprocess.check_output(['git','show',f'{source}:{name}'],cwd=ROOT)
  assert 'shared_contact.h' in archive.read('spatial_backend/coulomb.h').decode()
  assert archive.read(str((study/'plan.json').relative_to(ROOT)))==(study/'plan.json').read_bytes()
 baseline=json.loads((ROOT/plan['baseline_plan']).read_text());oldscenes=json.loads((ROOT/plan['baseline_scenes']).read_text());scenes=json.loads((directory/'scenes.json').read_text())
 for key in ['common','dt_s','trajectory_budget','physical_gates','reference_rule','scenes']:assert plan[key]==baseline[key]
 assert scenes==oldscenes and plan['contact_point_policy']=='shared'
 expected={f'{c["id"]}/reference_{i}.json' for c in plan['scenes'] for i in range(3)};histories=0;rejections=[];receipts={}
 with zipfile.ZipFile(directory/'traces.zip') as archive:
  assert set(archive.namelist())==expected
  for config in plan['scenes']:
   name=config['id'];scene=scenes[name]['scene'];runs={};physical={};eligible={}
   for i,fraction in enumerate(config['fractions']):
    lane=f'reference_{i}';key=f'{name}/{lane}.json';result=json.loads(archive.read(key));runs[lane]=result
    assert result==json.loads((directory/'checkpoints'/key).read_text())
    if 'rejected' in result:
     eligible[lane]=False;assert result['exit_code']==1 and result['elapsed_s']>0
     diagnostic=summary['rejection_diagnostics'][key];dump=directory/diagnostic['path'];assert sha(dump.read_bytes())==diagnostic['sha256']
     snapshot=json.loads(dump.read_text());assert snapshot['tolerance_m_s']==plan['common']['contact_tolerance_m_s'] and snapshot['residual_m_s']>snapshot['tolerance_m_s']
     if snapshot['phase']=='position':
      assert result['rejected'].startswith('Normal-only position projection residual failed')
     else:
      assert snapshot['phase']=='velocity' and result['rejected'].startswith('Coulomb residual gate failed') and 'no friction-law fallback' in result['rejected']
     assert snapshot['phase']==diagnostic['phase'] and len(snapshot['b'])==diagnostic['rows']
     A=np.asarray(snapshot['A']);assert A.shape==(len(snapshot['b']),)*2 and np.isfinite(A).all() and np.allclose(A,A.T,rtol=1e-12,atol=1e-12)
     rejections.append(dict(scene=name,lane=lane,fraction=fraction,rows=len(snapshot['b']),residual_m_s=snapshot['residual_m_s'],reason=result['rejected']))
    else:
     histories+=1;model=result['numerical_model'];assert model['contact_point_policy']==result['contact_point_policy']=='shared'
     for key2,value in plan['common'].items():
      if key2=='contact_recovery':assert model[key2]['enabled']==value
      else:assert model[key2]==value
     assert model['travel_fraction']==fraction and result['scalar_precision']=='float64'
     expected_times=np.arange(round(scene['duration']/plan['dt_s'])+1)*plan['dt_s'];assert np.allclose(result['times'],expected_times,rtol=0,atol=1e-12)
     d=trajectory_metrics(scene,result);physical[lane]=d;eligible[lane]=all(np.isfinite(d[k]) and d[k]<=limit for k,limit in plan['physical_gates'].items()) and result['coulomb_residual_max_m_s']<=plan['common']['contact_tolerance_m_s']
   edges=[]
   for left,right in [('reference_0','reference_1'),('reference_1','reference_2')]:
    error=trajectory_error(runs[left],runs[right]) if eligible.get(left) and eligible.get(right) else None
    passed=bool(error and all(error[k]<=limit/4 for k,limit in plan['trajectory_budget'].items()))
    edges.append(dict(left=left,right=right,error=error,passed=passed))
   receipt=summary['scenes'][name];assert receipt['reference_qualified']==all(e['passed'] for e in edges)
   assert receipt['choice'] is None and receipt['candidates']=={}
   for a,b in zip(edges,receipt['edges']):
    assert a['passed']==b['passed'] and (a['error'] is None)==(b['error'] is None)
    if a['error']:assert all(np.isclose(a['error'][k],b['error'][k],rtol=1e-9,atol=1e-10) for k in a['error'])
   receipts[name]=dict(reference_qualified=all(e['passed'] for e in edges),physical_eligible=eligible,edges=edges,diagnostics=physical)
 assert histories==summary['history_count']
 out=dict(execution_source_commit=source,attempt_count=6,history_count=histories,rejection_count=len(rejections),rejections=rejections,scenes=receipts)
 (study/'independent-audit.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print('Shared hull audit PASS:',histories,'histories;',len(rejections),'retained rejections;',sum(r['reference_qualified'] for r in receipts.values()),'qualified references')
 return out
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--directory',default=str(DIRECTORY));parser.add_argument('--source-commit',default=SOURCE);args=parser.parse_args();study=Path(args.directory)
 if not study.is_absolute():study=ROOT/study
 audit(study.resolve(),args.source_commit)
