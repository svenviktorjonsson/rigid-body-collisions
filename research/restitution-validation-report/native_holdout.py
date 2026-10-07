"""Native verification of every held-out prediction from all candidate laws."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from spatial_engine import run, BINARY, energy
from research.spatial_scenes import sphere
H=Path(__file__).resolve().parent;D=H/'native-heldout';D.mkdir(exist_ok=False);f=json.loads((H/'fits.json').read_text());records=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();guard=sha(BINARY)
for kind,model in f['models'].items():
 for item in model['heldout']['predictions']:
  obs=next(r for r in f['data'] if r['row']==item['source_row']);r=obs['diameter_m']/2;mass=1.2 if r<.075 else 10.;en,et,mu=item['coefficients']
  scene={'duration':1e-6,'gravity':[0,0,0],'bodies':[{'type':'kinematic','position':[0,0,-.1],'friction':1.,'shapes':[{'kind':'box','half_extents':[2,2,.1]}]},sphere([0,0,r-1e-12],radius=r,mass=mass,velocity=[obs['vt_before_m_s'],0,-obs['vn_before_m_s']],friction=mu)]}
  try:
   result=run(scene,dt=1e-6,primary_steps=1,iterations=4096,travel_fraction=0,solver='coulomb',kinematic_contact_phase='start',position_stabilization='split_translation_combined',normal_restitution=en,tangential_restitution=et,record_contact_impacts=True)
   s=np.array(result['states'])[-1,1];pred=np.array([s[9],s[7],abs(s[11])]);error=float(np.max(abs(pred-item['predicted'])));kinetic=float(energy(result)[-1]-energy(result)[0]-result['boundary_work_J'])
   check={'model':kind,'source_row':item['source_row'],'analytic_native_max_error':error,'kinetic_change_minus_boundary_work_J':kinetic,'native_law_check_passed':bool(error<1e-7 and kinetic<1e-8),'experimental_full_state_validated':False};entry={'scene':scene,'result':result,'heldout_target':item,'checks':check}
  except Exception as e:
   check={'model':kind,'source_row':item['source_row'],'native_law_check_passed':False,'error':repr(e),'experimental_full_state_validated':False};entry={'scene':scene,'heldout_target':item,'checks':check}
  records.append(check);(D/f'{kind}_row{item["source_row"]}.json').write_text(json.dumps(entry,indent=2,allow_nan=False)+'\n')
assert guard==sha(BINARY)
summary={'case_count':len(records),'native_law_pass_count':sum(r['native_law_check_passed'] for r in records),'records':records,'binary_sha256':guard,'all_native_predictions_reproduced':all(r['native_law_check_passed'] for r in records),'actual_specimen_geometry_validated':False}
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Native heldout checks',summary['native_law_pass_count'],'/',len(records),flush=True)
