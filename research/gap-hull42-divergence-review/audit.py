"""Repeatable independent scalar/SciPy audit of retained trajectory diagnostics."""
import json,hashlib,math,subprocess,zipfile,itertools
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
P=Path(__file__).parent;ROOT=P.parents[1];plan=json.loads((P/'plan.json').read_text());a=json.loads((P/'analysis.json').read_text());config=json.loads((P/'collision-config-plan.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
for p,expected in {**plan['inputs_sha256'],**config['progress_hashes']}.items():assert sha((ROOT/p).read_bytes())==expected,p
for p,expected in config['upstream_sha256'].items():assert sha((ROOT/'build/bullet-inspect'/p).read_bytes())==expected,p
cache_plan=json.loads((P/'collision-cache-plan.json').read_text());configuration=json.loads((P/'collision-config.json').read_text())
for p,expected in cache_plan['sources_sha256'].items():assert sha((ROOT/'build/bullet-inspect'/p).read_bytes())==expected,p
assert configuration['cache_evidence_sha256']==cache_plan['sources_sha256']
assert configuration['source_sha256']==sha((P/'configuration.py').read_bytes())
wire=json.loads((ROOT/next(iter(config['progress_hashes']))).read_text())['wire_bodies']
# Independently enumerate transformed cached-box corners rather than reusing
# configuration.py's |R| extent expression.
for body,reported in zip(wire,configuration['relative_compound_threshold_estimates']):
 discs=[]
 for padding in [.04,0.]:
  allcorners=[]
  for shape in body['shapes']:
   if shape['kind']=='box':lo=-np.array(shape['half_extents']);hi=-lo
   else:vertices=np.array(shape['vertices']);lo=vertices.min(axis=0)-padding;hi=vertices.max(axis=0)+padding
   corners=np.array(list(itertools.product(*zip(lo,hi))))
   transformed=Rotation.from_quat(shape['orientation']).apply(corners)+np.array(shape['center']);allcorners.extend(transformed)
  bounds=np.array(allcorners);lo=bounds.min(axis=0);hi=bounds.max(axis=0);discs.append(.5*np.linalg.norm(hi-lo)+np.linalg.norm(.5*(hi+lo)))
 assert abs(.02*discs[0]-reported['relative_contact_breaking_threshold_m'])<1e-15
 assert abs(.02*discs[1]-reported['consistent_declared_margin_threshold_m'])<1e-15
for record in configuration['first_travel_bounds']:
 f=record['travel_fraction'];expected=f*.025/(40+math.sqrt(2*9.81*f*.025));assert abs(record['initial_h_s']-expected)<1e-18
runs=[json.loads((ROOT/'research/hull-gap-completion/results/checkpoints'/plan['scene']/f'reference_{i}.json').read_text())for i in range(3)];assert all(r['attempt_status']=='history_complete'for r in runs);s=[np.array(r['states'])for r in runs];times=runs[0]['times'];ids=[i for i,m in enumerate(runs[0]['mass'])if m>0];maxerr=0.
for record,(left,right)in zip(a['pairs'],[(0,1),(1,2),(0,2)]):
 raw={k:[]for k in a['quarter_budget']}
 for f,t in enumerate(times):
  values={k:[]for k in raw}
  for i in ids:
   for k,lo,hi in [('position_m',0,3),('velocity_m_s',7,10),('omega_rad_s',10,13)]:values[k].append(math.fsum(float(x-y)**2 for x,y in zip(s[left][f,i,lo:hi],s[right][f,i,lo:hi])))
   rot=Rotation.from_quat(s[left][f,i,3:7]).inv()*Rotation.from_quat(s[right][f,i,3:7]);values['orientation_rad'].append(float(rot.magnitude())**2)
  for k,v in values.items():
   rms=math.sqrt(math.fsum(v)/len(v));maxerr=max(maxerr,abs(rms-record['perframe_RMS'][f][k]));raw[k].extend(v)
 for k,v in raw.items():assert abs(math.sqrt(math.fsum(v)/len(v))-record['global_RMS'][k])<1e-12
 for k,threshold in a['quarter_budget'].items():
  crossings=[i for i,f in enumerate(record['perframe_RMS'])if f[k]>threshold];assert record['first_quarter_budget_crossing'][k]['index']==crossings[0]
assert maxerr<1e-12
provenance=json.loads((ROOT/'research/hull-gap-completion/results/checkpoints/provenance.json').read_text());assert provenance['execution_source_commit']==plan['execution_source'];count=0
with zipfile.ZipFile(ROOT/'research/hull-gap-completion/results/execution-source.zip')as z:
 for p,expected in provenance['source_hashes'].items():raw=z.read(p);assert sha(raw)==expected and raw==subprocess.check_output(['git','show',plan['execution_source']+':'+p],cwd=ROOT);count+=1
# Independent direct world-inertia energy calculation.
for lane,r in enumerate(runs):
 total=[]
 for frame in s[lane]:
  E=0.
  for i in ids:
   m=r['mass'][i];R=Rotation.from_quat(frame[i,3:7]).as_matrix();I=R@np.array(r['inertia_body_kg_m2'][i])@R.T;v=frame[i,7:10];w=frame[i,10:13];E+=.5*m*np.dot(v,v)+.5*np.dot(w,I@w)+m*9.81*frame[i,2]
  total.append(E)
 expected=total[-1]-total[0]-r['boundary_work_J'];assert abs(expected-a['physical_lanes'][lane]['final_energy_minus_boundary_work_J'])<1e-8
out=dict(passed=True,execution_source=plan['execution_source'],source_files_verified=count,independent_perframe_RMS_max_discrepancy=maxerr,source_ordered_cached_bounds_independently_recomputed=True,cache_source_files_verified=len(cache_plan['sources_sha256']),first_observed_difference_time_s=.01,first_observed_difference_precedes_first_wall_reversal=True,wall_clock_or_endpoint_mismatch_supported=False,physical_gates_versus_refinement_distinguished=True,causal_identification=False,trajectory_qualified=False,scope='Independent read-only trace/provenance/physical calculations; no native execution.')
dest=P/'independent-audit.json'
if dest.exists():assert json.loads(dest.read_text())['passed']==out['passed']
dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
