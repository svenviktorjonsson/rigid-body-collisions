"""Independent impulse/energy audit of native diagnostic; retain real mismatch."""
import json
import math
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
s=json.loads((H/'results/summary.json').read_text())
a=math.radians(25);m=.103;r=.029;v=4.;I=.4*m*r*r;vx=v*math.sin(a);vn=v*math.cos(a)
records=[]
for record in s['records']:
 mu=record['pair_friction_hypothesis']
 d=json.loads((H/'results'/f'mu{mu}.json').read_text());states=np.array(d['result']['states']);initial,last=states[0,1],states[-1,1]
 jn=m*(1+.78)*vn;jt=-min(mu*jn,vx/(1/m+r*r/I))
 expected=np.array([vx+jt/m,0,.78*vn,0,-r*jt/I,0])
 error=float(np.max(abs(last[7:13]-expected)))
 energy=lambda state:.5*m*float(state[7:10]@state[7:10])+.5*I*float(state[10:13]@state[10:13])
 change=energy(last)-energy(initial)
 assert error<1e-8 and change<=1e-10 and abs(d['result']['boundary_work_J'])<1e-12
 records.append({'mu':mu,'independent_velocity_spin_error':error,'ball_kinetic_change_J':change,'rigid_law_check_passed':True})
assert s['max_spin_factor_over_angle_uncertainty_rad_per_m']<s['measured_min_spin_factor_rad_per_m']
assert not s['model_reproduces_measured_spin'] and not s['independent_material_validation_passed']
result={'native_rigid_law_checks_passed':True,'records':records,'experimental_spin_match':False,'independent_material_validation_passed':False,'conclusion':'Native solver matches its rigid impulse law; that law fails the published rubber spin result under the stated geometry/inertia/support assumptions, even allowing reported angle uncertainty.'}
(H/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('Native impulse/energy audit PASS; experimental rubber spin mismatch CONFIRMED')
