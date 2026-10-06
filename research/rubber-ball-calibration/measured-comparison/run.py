"""Native rigid-law diagnostic against a public rubber-impact summary.
No parameter fits: unknown friction is bracketed; restitution is imposed from the
same experiment and must never be counted as an independently predicted match.
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from spatial_engine import run, BINARY
from research.spatial_scenes import sphere

H=Path(__file__).resolve().parent
D=H/'results';D.mkdir(exist_ok=False)
save=lambda p,d:p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
source='https://physics.usyd.edu.au/~cross/PUBLICATIONS/48.%20EnhanceBounce.pdf'
radius,mass,speed,angle=.029,.103,4.,math.radians(25.)
vx,vz=speed*math.sin(angle),-speed*math.cos(angle)
e_n,e_t,spin_factor=.78,.49,14.9
# Approximate speed is only given as "about 4m/s" in the paper. Ratio-based
# spin comparison avoids treating it as an exact per-trial velocity measurement.
observed_spin=spin_factor*speed
rigid_max_spin_factor=math.sin(angle)/(1.4*radius)
observed={'source':source,'table':'I','diameter_m':2*radius,'mass_kg':mass,
 'incident_speed_m_s_approximate':speed,'incident_angle_deg_to_vertical':25,
 'incident_angle_uncertainty_deg':1,'normal_restitution':e_n,
 'tangential_restitution':e_t,'normal_and_tangential_restitution_error':.01,
 'outgoing_spin_factor_rad_per_m':spin_factor,'spin_factor_error_rad_per_m':.1,
 'independently_measured_pair_friction':None,
 'inertia_status':'homogeneous solid-sphere assumption; independent measurement absent',
 'rigid_wall_status':'fixed-plane approximation to reported14kg granite block'}
save(D/'observation.json',observed)
receipt={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'binary':str(BINARY),'binary_sha256':hashlib.sha256(BINARY.read_bytes()).hexdigest(),
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
records=[]
for mu in (0.,.05,.1,.2,.4,.9,1.5):
 wall={'type':'kinematic','position':[0,0,-.1],'friction':1.,'restitution':1.,'shapes':[{'kind':'box','half_extents':[2,2,.1]}]}
 body=sphere([0,0,radius-1e-12],radius=radius,mass=mass,velocity=[vx,0,vz],friction=mu,restitution=e_n)
 scene={'duration':1e-5,'gravity':[0,0,0],'bodies':[wall,body]}
 result=run(scene,dt=1e-5,primary_steps=1,iterations=4096,travel_fraction=0,solver='coupled',kinematic_contact_phase='end',position_stabilization='split')
 state=np.asarray(result['states'])[-1,1];spin=state[11];u_after=state[7]-radius*spin
 predicted={'normal_restitution':float(-state[9]/vz),'tangential_restitution':float(-u_after/vx),'spin_factor_rad_per_m':float(spin/speed)}
 record={'pair_friction_hypothesis':mu,'friction_source':'diagnostic sweep; no independent same-pair measurement',
 'normal_restitution_source':'imposed from this same impact; not an independent prediction',
 'predicted':predicted,'errors':{k:predicted[k]-observed[k] for k in predicted},
 'spin_within_reported_error':bool(abs(predicted['spin_factor_rad_per_m']-spin_factor)<=.1),
 'independent_material_validation_passed':False}
 save(D/f'mu{mu}.json',{'scene':scene,'result':result,'comparison':record});records.append(record)
 summary_spin=predicted['spin_factor_rad_per_m'];print(mu,predicted,flush=True)
summary={'observed':observed,'records':records,'native_runs_completed':len(records),
 'rigid_point_contact_max_spin_factor_rad_per_m':rigid_max_spin_factor,
 'max_spin_factor_over_angle_uncertainty_rad_per_m':math.sin(math.radians(26))/(1.4*radius),
 'measured_min_spin_factor_rad_per_m':spin_factor-.1,
 'model_reproduces_measured_spin':any(r['spin_within_reported_error'] for r in records),
 'independent_material_validation_passed':False,
 'parameter_fit_performed':False,'provenance':receipt,
 'limitations':['Missing independently measured rubber/granite friction.',
                'Normal restitution supplied from the comparison experiment.',
                'No tangential compliance, finite-area traction or contact-duration model.',
                'Fixed support approximation, homogeneous sphere inertia and approximate incident speed.']}
assert hashlib.sha256(BINARY.read_bytes()).hexdigest()==receipt['binary_sha256']
save(D/'summary.json',summary)
