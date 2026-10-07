"""Frozen standalone signed-gap position-law trial and exact geometry re-query."""
import argparse,hashlib,json,os,re,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,least_squares
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).parent
sha=lambda data:hashlib.sha256(data).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-commit');ap.add_argument('--check-plan',action='store_true');a=ap.parse_args();plan=json.loads((P/'plan.json').read_text());cap=ROOT/plan['capture'];geo=ROOT/plan['geometry']
 assert sha(cap.read_bytes())==plan['capture_sha256'] and sha(geo.read_bytes())==plan['geometry_sha256']
 d=json.loads(cap.read_text());g=json.loads(geo.read_text());A=np.array(d['A']);oldb=np.array(d['b']);hi=np.array(d['hi']);h=d['internal_dt_s'];tol=plan['tolerance_m_s'];slop=plan['contact_slop_m'];assert d['tolerance_m_s']==tol and all(x==-1 for x in d['dependencies']) and all(x==0 for x in d['lo'])
 distance=np.array([r['signed_distance_m']for r in g['rows']]);b=np.where(distance>slop,-(distance-slop)/h,oldb);assert all(oldb[abs(distance)<=slop]==0)
 if a.check_plan:print('Plan valid: unchanged74 A/bounds/material; only declared separated POSITION targets change; no execution');return
 if not a.source_commit:ap.error('--source-commit mandatory; freeze before execution')
 source=subprocess.check_output(['git','rev-parse',a.source_commit],cwd=ROOT,text=True).strip();paths=[str((P/f).relative_to(ROOT))for f in ['plan.json','runner.py']]+['research/translation-position-geometry-review/requery.cpp']
 def guard_sources():
  for p in paths:assert (ROOT/p).read_bytes()==subprocess.check_output(['git','show',source+':'+p],cwd=ROOT),p
  assert sha(cap.read_bytes())==plan['capture_sha256'] and sha(geo.read_bytes())==plan['geometry_sha256']
 guard_sources();results=P/'results';results.mkdir(exist_ok=False)
 binary=results/'geometry_requery';bullet=ROOT/'build/bullet-inspect/src';static=[ROOT/'build/spatial/_deps/bullet-build/src'/v for v in ['BulletDynamics/libBulletDynamics.a','BulletCollision/libBulletCollision.a','LinearMath/libLinearMath.a']]
 command=['c++','-std=c++17','-O2','-Wall','-Wextra','-Wpedantic','-DBT_USE_DOUBLE_PRECISION','-I'+str(bullet),'-I'+str(ROOT/'build/spatial/_deps/json-src/single_include'),str(ROOT/paths[-1])]+[str(x)for x in static]+['-o',str(binary)]
 subprocess.run(command,check=True,cwd=ROOT);binary_hash=sha(binary.read_bytes());ldd=subprocess.check_output(['ldd',str(binary)],text=True);libraries={str(Path(v).resolve()):sha(Path(v).read_bytes())for v in re.findall(r'^\s*\S+\s+=>\s+(/\S+)',ldd,re.M)};libhash={str(x):sha(x.read_bytes())for x in static}
 def guard():
  guard_sources();assert sha(binary.read_bytes())==binary_hash
  for path,expected in {**libraries,**libhash}.items():assert sha(Path(path).read_bytes())==expected,path
 provenance=dict(execution_source_commit=source,source_hashes={p:sha((ROOT/p).read_bytes())for p in paths},capture_sha256=plan['capture_sha256'],geometry_sha256=plan['geometry_sha256'],compiler_command=command,compiler_version=subprocess.check_output(['c++','--version'],text=True),binary_sha256=binary_hash,runtime_library_hashes=libraries,static_library_hashes=libhash,numpy_version=np.__version__)
 with zipfile.ZipFile(results/'source.zip','w',zipfile.ZIP_DEFLATED)as z:
  for p in paths:z.write(ROOT/p,p)
  z.write(cap,plan['capture']);z.write(geo,plan['geometry'])
 (results/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 bodies={q['solver_body_id']:q for q in g['bodies']};finite=[i for i,q in bodies.items()if q['inverse_mass']>0];indices={v:k for k,v in enumerate(finite)};n=len(b);J=np.zeros((n,3*len(finite)));M=np.zeros((3*len(finite),3*len(finite)))
 for k,i in enumerate(finite):M[3*k:3*k+3,3*k:3*k+3]=np.eye(3)*bodies[i]['inverse_mass']
 for i,r in enumerate(g['rows']):
  for side in ['a','b']:
   j=r['solver_body_id_'+side]
   if j in indices:k=indices[j];J[i,3*k:3*k+3]=r['linear_jacobian_'+side]
 assert np.max(abs(J@M@J.T-A))<1e-12
 rho=1/np.diag(A)
 def gate(p,target):
  w=A@p-target;res=float(np.max(abs(p-np.maximum(0,p-rho*w))/rho));energy=float(.5*p@A@p-target@p);scale=float(1+np.sum(abs(p*target)));finite_ok=bool(np.isfinite(p).all()and np.isfinite(w).all()and np.isfinite(energy)and np.isfinite(scale));return dict(accepted=finite_ok and np.min(p)>=0 and np.all(p<=hi)and res<=tol and np.min(w)>=-tol and energy<=tol*scale,residual_m_s=res,min_normal_velocity_m_s=float(np.min(w)),passive_change_bound_J=energy,pressure_max_Ns=float(np.max(p)))
 def fb(p,jac=False):
  u=p/rho;w=A@p-b;l=np.hypot(u,w)
  if jac:
   ca=1-np.divide(u,l,out=np.zeros_like(u),where=l>0);cb=1-np.divide(w,l,out=np.zeros_like(w),where=l>0);Q=cb[:,None]*A;Q[np.diag_indices_from(Q)]+=ca/rho;return Q
  return u+w-l
 gravity=np.array(plan['gravity_m_s2']);report=dict(schema=plan['schema'],original_target=oldb.tolist(),prospective_target=b.tolist(),finite_body_ids=[bodies[i]['body_id']for i in finite],attempts=[],original_policy_qualified=False)
 for trial in plan['trials']:
  guard();method=trial['method'];start=time.perf_counter();p=np.zeros(n)
  if method=='pressure-QP-cold':r=minimize(lambda p:.5*p@A@p-b@p,p,jac=lambda p:A@p-b,bounds=[(0,None)]*n,method='SLSQP',options={'ftol':1e-24,'maxiter':trial['max_iterations']})
  else:r=least_squares(fb,p,jac=lambda p:fb(p,True),max_nfev=trial['max_evaluations'],ftol=1e-14,xtol=1e-14,gtol=1e-14)
  p=np.maximum(r.x,0);delta=(h*M@J.T@p).reshape(-1,3);vec=np.zeros((len(finite),6));vec[:,:3]=delta;masses=np.array([bodies[i]['mass']for i in finite]);velocity=np.array([bodies[i]['linear_velocity']for i in finite]);spin=np.array([bodies[i]['angular_velocity']for i in finite]);centers=np.array([bodies[i]['world_transform']['position']for i in finite]);inertia=np.array([np.linalg.inv(bodies[i]['inverse_world_inertia'])for i in finite]);linear_before=np.sum(masses[:,None]*velocity,axis=0);orbital_before=np.sum(np.cross(centers,masses[:,None]*velocity),axis=0);spin_before=np.sum(np.einsum('ijk,ik->ij',inertia,spin),axis=0);kinetic=float(.5*np.sum(masses*np.sum(velocity**2,axis=1))+.5*np.einsum('ni,nij,nj',spin,inertia,spin));deltaL=np.sum(np.cross(delta,masses[:,None]*velocity),axis=0);delta_com=np.sum(masses[:,None]*delta,axis=0)/masses.sum();gravity_change=float(-np.sum(masses*(delta@gravity)))
  record=dict(method=method,optimizer_success=bool(r.success),message=str(r.message),nfev=int(r.nfev),iterations=int(getattr(r,'nit',0)),elapsed_s=time.perf_counter()-start,raw_impulse=r.x.tolist(),impulse=p.tolist(),original_gate=gate(p,oldb),prospective_gate=gate(p,b),pose_increment=vec.tolist(),max_translation_norm_m=float(np.max(np.linalg.norm(delta,axis=1))),max_rotation_norm_rad=0.,linear_qualified=bool(gate(p,b)['accepted']),trajectory_qualified=False,ledger=dict(scope='Captured rigid-body velocities before physical MLCP impulse application; pose effect only, not a completed collision ledger',total_mass_kg=float(masses.sum()),COM_change_m=delta_com.tolist(),linear_momentum_before_kg_m_s=linear_before.tolist(),linear_momentum_change_kg_m_s=[0.,0.,0.],angular_momentum_before_kg_m2_s=(orbital_before+spin_before).tolist(),total_angular_momentum_change_kg_m2_s=deltaL.tolist(),spin_angular_momentum_change_kg_m2_s=[0.,0.,0.],kinetic_energy_before_J=kinetic,kinetic_energy_change_J=0.,gravity_potential_change_J=gravity_change,numerical_pose_work_J=gravity_change,boundary_physical_work_added_J=0.))
  report['attempts'].append(record);(results/'position-guides.json').write_text(json.dumps(report,indent=2)+'\n');guard();print(method,record['prospective_gate'],flush=True)
 with open(results/'requery.stdout','w')as log:subprocess.run([str(binary),str(geo),str(results/'position-guides.json'),str(results/'geometry-requery.json')],stdout=log,check=True,cwd=ROOT)
 query=json.loads((results/'geometry-requery.json').read_text())
 def pair_min(q):
  out={}
  for c in q['contacts']:
   pair=tuple(sorted([c['body_a'],c['body_b']]));out[pair]=min(out.get(pair,float('inf')),c['signed_distance_m'])
  return out
 before=pair_min(query['before'])
 for record,requery in zip(report['attempts'],query['trials']):
  after=pair_min(requery['after']);excess=max([0.]+[min(before.get(pair,0.),0.)-v for pair,v in after.items()]);record['geometry_requery']=dict(max_penetration_before_m=query['before']['max_penetration_m'],max_penetration_after_m=requery['after']['max_penetration_m'],largest_pair_gap_worsening_m=excess,within_original_contact_slop=excess<=slop);record['prospective_trial_qualified']=bool(record['linear_qualified']and record['max_translation_norm_m']<=plan['pose_bound_m']and excess<=slop)
 guard();report['prospective_qualified']=any(t['prospective_trial_qualified']for t in report['attempts']);(results/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('PROSPECTIVE_TRIAL',report['prospective_qualified'],flush=True)
if __name__=='__main__':main()
