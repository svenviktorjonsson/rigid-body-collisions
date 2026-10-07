"""Independent time integration checks of the declared linear wrench hypothesis."""
from pathlib import Path
import json,math
import numpy as np
from scipy.integrate import solve_ivp
P=Path(__file__).resolve().parent;s=json.loads((P/'results.json').read_text());checks=[]
for c in s['controls']+s['folds']:
 k=c['shear_stiffness'];b=c['rocking_stiffness'];a=.4
 def ode(t,z):
  x,theta,v,w=z;F=-k*(x-theta);C=-b*theta
  return [v,w,F,(-F+C)/a]
 errors=[];energy_errors=[]
 for step in [1/64,1/128]:
  z=solve_ivp(ode,[0,1],[0,0,1,0],rtol=1e-11,atol=1e-13,max_step=step,dense_output=True)
  x,theta,v,w=z.y[:,-1];errors.append(max(abs(v-c['vx_ratio']),abs(w-c['omega_R_over_incoming_tangent_speed'])))
  X,T,V,W=z.sol(np.linspace(0,1,257));E=.5*(V*V+a*W*W+k*(X-T)**2+b*T*T);energy_errors.append(float(np.max(abs(E-.5))))
 assert max(errors)<1e-8 and max(energy_errors)<1e-8
 checks.append({'k':k,'b':b,'max_step_endpoint_errors':errors,'history_energy_errors':energy_errors,'passed':True})
result={'case_count':len(checks),'pass_count':len(checks),'records':checks,'independent_solver':'scipy solve_ivp RK45; two max-step limits; analytic result from mass-scaled spectral solution','physical_patch_admissibility_checked':False,'experimental_improvement':False}
(P/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('independent integration/energy cases',len(checks),'PASS')
