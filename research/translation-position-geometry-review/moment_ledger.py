"""Independent captured-state effect of research-only angular pose guides."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
P=Path(__file__).parent;D=P/'results';dest=D/'pose-moment-energy-ledger.json';assert not dest.exists()
paths=[D/'rejected-normal-system.json.geometry.json',D/'linear-pose-guides.json'];g,guide=[json.loads(p.read_text())for p in paths];bodies={b['body_id']:b for b in g['bodies']if b['has_original_body']};gravity=np.array([0,0,-9.81]);out=dict(schema='captured-state-pose-guide-ledger-v1',scope='Independent pose change at frozen captured rigid velocities; physical MLCP impulse not yet applied. No pose guide integrated.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},trials=[])
for trial in guide['attempts']:
 mass=0.;COM=np.zeros(3);Pchange=np.zeros(3);Lchange=np.zeros(3);Lorb=np.zeros(3);Lspin=np.zeros(3);KE=0.;PE=0.;spin_norm=0.
 for ident,d in zip(guide['finite_body_ids'],trial['pose_increment']):
  b=bodies[ident];m=b['mass'];dx=np.array(d[:3]);theta=np.array(d[3:]);v=np.array(b['linear_velocity']);w=np.array(b['angular_velocity']);I=np.linalg.inv(b['inverse_world_inertia']);R=Rotation.from_rotvec(theta).as_matrix();newI=R@I@R.T;deltaI=newI-I;delta_spin=deltaI@w;delta_orb=np.cross(dx,m*v);mass+=m;COM+=m*dx;Lspin+=delta_spin;Lorb+=delta_orb;Lchange+=delta_spin+delta_orb;KE+=.5*w@deltaI@w;PE-=m*gravity@dx;spin_norm=max(spin_norm,float(np.linalg.norm(delta_spin)))
 out['trials'].append(dict(method=trial['method'],physical_velocity_change=[0.,0.,0.],COM_change_m=(COM/mass).tolist(),linear_momentum_change_kg_m_s=Pchange.tolist(),spin_angular_momentum_change_kg_m2_s=Lspin.tolist(),orbital_angular_momentum_change_kg_m2_s=Lorb.tolist(),total_angular_momentum_change_kg_m2_s=Lchange.tolist(),rotational_kinetic_energy_change_J=float(KE),gravity_potential_change_J=float(PE),unaccounted_pose_energy_change_J=float(KE+PE),physically_integrated=False))
dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
