"""Passive finite-duration shear/rocking hypothesis; not production physics."""
from pathlib import Path
import json, math, hashlib
import numpy as np
from scipy.optimize import brentq
P=Path(__file__).resolve().parent
raw=P.parent/'restitution-validation-report/rubber-surfaces/summary.json'
rows=json.loads(raw.read_text())['rows'];alpha=.4;R=.029;angle=math.radians(25)
inv=np.diag([1.,1/math.sqrt(alpha)])
v0=np.array([1.,0.]);z0=np.array([1.,0.])
def endpoint(k,b):
 K=np.array([[k,-k],[-k,k+b]])
 vals,V=np.linalg.eigh(inv@K@inv);w=np.sqrt(np.maximum(vals,0))
 q=inv@V@(np.sinc(w/math.pi)*(V.T@z0));v=inv@V@(np.cos(w)*(V.T@z0))
 kinetic=.5*(v[0]**2+alpha*v[1]**2);strain=float(.5*q@K@q)
 return q,v,kinetic,strain

def conditional(et,b):
 # First root in ascending shear stiffness: branch chosen without spin data.
 def residual(k):
  _,v,_,_=endpoint(k,b);return v[1]-v[0]-et
 mesh=np.linspace(0,40,161);prev=residual(0)
 for lo,hi in zip(mesh[:-1],mesh[1:]):
  nxt=residual(hi)
  if prev*nxt<=0:
   k=brentq(residual,lo,hi,xtol=1e-12);q,v,E,U=endpoint(k,b)
   return {'shear_stiffness':k,'rocking_stiffness':float(b),'vx_ratio':float(v[0]),'omega_R_over_incoming_tangent_speed':float(v[1]),'spin_factor':float(v[1]*math.sin(angle)/R),'normalised_kinetic_energy':E,'separation_strain_energy_dissipated':U,'energy_balance_error':E+U-.5,'restitution_error':float(v[1]-v[0]-et),'tangent_impulse':float(v[0]-1),'rocking_moment_impulse':float(alpha*v[1]+v[0]-1)}
  prev=nxt
 return None
bs=np.linspace(0,20,161);catalog=[]
for b in bs:
 cases=[conditional(row['tangential_restitution'],b) for row in rows]
 catalog.append({'b':float(b),'cases':cases})
folds=[]
for held in range(4):
 valid=[c for c in catalog if all(c['cases'][j] is not None for j in range(4))]
 best=min(valid,key=lambda c:sum((c['cases'][j]['spin_factor']-rows[j]['observed_spin_factor'])**2 for j in range(4) if j!=held))
 case=best['cases'][held].copy();row=rows[held];pt=case['tangent_impulse'];C=case['rocking_moment_impulse'];normal=(1+row['normal_restitution'])/math.tan(angle)
 budget=math.hypot(pt,C/.3);case.update({'surface':row['surface'],'observed_spin_factor':row['observed_spin_factor'],'training_surfaces':[r['surface'] for j,r in enumerate(rows) if j!=held],'baseline_spin_factor':row['predicted_central_spin_factor'],'integrated_wrench_budget_pass':budget<=.9*normal,'required_integrated_friction':budget/normal,'instantaneous_patch_admissibility_validated':False})
 folds.append(case)
baseline=math.sqrt(sum((r['predicted_central_spin_factor']-r['observed_spin_factor'])**2 for r in rows)/4)
candidate=math.sqrt(sum((c['spin_factor']-c['observed_spin_factor'])**2 for c in folds)/4)
controls=[]
for et in [-.8,-.1,0,.49,.57,.9]:
 for b in [0,.125,1,5,10,20]:
  c=conditional(et,b)
  if c is not None:
   assert abs(c['energy_balance_error'])<1e-10 and abs(c['restitution_error'])<1e-9
   assert c['normalised_kinetic_energy']<=.5+1e-10
   if b==0:assert abs(c['omega_R_over_incoming_tangent_speed']-(1+et)/1.4)<1e-10
   controls.append(c)
summary={'source_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'baseline_heldout_spin_rmse_rad_m':baseline,'candidate_heldout_spin_rmse_rad_m':candidate,'relative_rmse_change':candidate/baseline-1,'folds':folds,'control_pass_count':len(controls),'controls':controls,'shared_parameter_grid':[0,20,161],'all_folds_integrated_budget_pass':all(c['integrated_wrench_budget_pass'] for c in folds),'production_adopted':False,'real_patch_validated':False,'separation_loss_is_explicit_model_assumption':True,'interpretation':'Conditional restitution supplied from target experiment; shared rocking parameter fit on other surfaces. Lowest shear root fixed without spin fitting. Passive energy accounting alone does not prove real contact geometry.'}
(P/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
(P/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
print('baseline RMSE',baseline,'candidate RMSE',candidate,'controls',len(controls))
for c in folds:print(c['surface'],'b',c['rocking_stiffness'],'spin',c['spin_factor'],'budget',c['integrated_wrench_budget_pass'])
