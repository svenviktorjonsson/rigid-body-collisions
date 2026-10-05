"""Predeclared hybrid elastic-contact completion; checkpoint every attempt."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.elastic_patch import Material,Plane,simulate
from research.elastic_impulse import resolve_impact
DIRECTORY=Path(__file__).resolve().parent
SOURCES=['research/elastic_patch.py','research/elastic_impulse.py','tests/test_elastic_patch.py','tests/test_elastic_impulse.py',
         'research/elastic-completion/runner.py','research/elastic-completion/plan.json',
         'research/elastic-patch/plan.json','research/elastic-patch-refined/plan.json']
EXTRAS=['expected_velocity','expected_omega','expected_dissipation','minimum_bounces','require_spin_reversal','step_scale','retain_rejection','bounce_pattern']

def serial(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
 if isinstance(value,list):return [serial(v) for v in value]
 return value

def canonical(value):return json.dumps(serial(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(data):return hashlib.sha256(data).hexdigest()
def atomic(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(data);temp.replace(path)

def guard(source):
 for path in SOURCES:
  if subprocess.check_output(['git','show',f'{source}:{path}'],cwd=ROOT)!=(ROOT/path).read_bytes():raise RuntimeError('Frozen source changed: '+path)

def metrics(result):
 initial=float(result['kinetic_J'][0]+result['potential_J'][0]+result['stored_J'][0])
 return dict(final_velocity_m_s=result['states'][-1,3:6].tolist(),final_omega_rad_s=result['states'][-1,6:9].tolist(),
  initial_energy_J=initial,max_energy_residual_J=float(max(abs(result['energy_residual_J']))),
  final_linear_impulse_N_s=result['linear_impulse_N_s'][-1].tolist(),final_couple_impulse_N_m_s=result['couple_impulse_N_m_s'][-1].tolist(),
  dissipated_J=float(result['dissipated_J'][-1]),max_yield_excess_N=float(result['max_yield_excess_N']),
  separation_loss_J=float(sum(e['separation_loss_J'] for e in result['events'])),
  rhs_evaluations=result['rhs_evaluations'],free_flight_segments=result['free_flight_segments'],
  events=result['events'],material_events=result['material_events'])

def qualify(case,levels,attempts,arrays,gates):
 checks={};edges=[]
 if all(r is not None for r in arrays):
  checks['energy']=all(a['metrics']['max_energy_residual_J']<=gates['energy_absolute_J']+gates['energy_relative']*abs(a['metrics']['initial_energy_J']) for a in attempts)
  checks['yield']=all(a['metrics']['max_yield_excess_N']<=gates['yield_excess_N'] for a in attempts)
  checks['nonnegative_dissipation']=all(np.min(r['dissipated_J'])>=-gates['energy_absolute_J'] and np.min(np.diff(r['dissipated_J']))>=-gates['energy_absolute_J'] for r in arrays)
  for a,b in zip(arrays,arrays[1:]):
   assert np.array_equal(a['times'],b['times']);difference=a['states']-b['states']
   edges.append(dict(position_m=float(np.max(np.linalg.norm(difference[:,:3],axis=1))),velocity_m_s=float(np.max(np.linalg.norm(difference[:,3:6],axis=1))),omega_rad_s=float(np.max(np.linalg.norm(difference[:,6:9],axis=1)))))
  checks['refinement']=all(e['position_m']<=gates['position_refinement_m'] and e['velocity_m_s']<=gates['velocity_refinement_m_s'] and e['omega_rad_s']<=gates['omega_refinement_rad_s'] for e in edges)
  sim=case['simulation'];fine=arrays[-1];met=attempts[-1]['metrics']
  if 'expected_velocity' in sim:checks['analytic_velocity']=np.linalg.norm(fine['states'][-1,3:6]-sim['expected_velocity'])<=gates['analytic_velocity_m_s']
  if 'expected_omega' in sim:checks['analytic_omega']=np.linalg.norm(fine['states'][-1,6:9]-sim['expected_omega'])<=gates['analytic_omega_rad_s']
  if 'expected_dissipation' in sim:checks['analytic_dissipation']=abs(met['dissipated_J']-sim['expected_dissipation'])<=1e-5
  if sim.get('require_spin_reversal'):checks['spin_reversal']=fine['states'][-1,8]*sim['omega'][2]<0
  checks['separation_loss']=all(a['metrics']['separation_loss_J']<=gates['separation_loss_J'] for a in attempts)
  if 'minimum_bounces' in sim:
   lifts=[e for e in met['events'] if e['kind']=='lift_off'];checks['bounces']=len(lifts)>=sim['minimum_bounces']
   if sim.get('bounce_pattern')=='same-floor-oblique':
    checks['back_and_forth']=all(e['plane']=='floor' and abs(e['velocity_m_s'][0]+sim['velocity'][0]*(-1)**i)<=gates['analytic_velocity_m_s'] and
     abs(e['omega_rad_s'][1]+sim['omega'][1]*(-1)**i)<=gates['analytic_omega_rad_s'] and abs(e['omega_rad_s'][2]+sim['omega'][2]*(-1)**i)<=gates['analytic_omega_rad_s'] for i,e in enumerate(lifts))
    checks['fine_force_capacity']=met['max_yield_excess_N']<=1e-5+1e-8*float(np.max(fine['normal_force_N']))
   else:
    checks['chained_floor_ceiling']=all(e['plane']==('floor' if i%2==0 else 'ceiling') and abs(e['velocity_m_s'][2]-abs(sim['velocity'][2])*(-1)**i)<=gates['analytic_velocity_m_s'] and
     abs(e['omega_rad_s'][2]+sim['omega'][2]*(-1)**i)<=gates['analytic_omega_rad_s'] for i,e in enumerate(lifts))
 return dict(checks={k:bool(v) for k,v in checks.items()},qualified=bool(checks and all(checks.values())),refinement_edges=edges)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--source-commit');parser.add_argument('--check-plan',action='store_true');args=parser.parse_args()
 plan=json.loads((DIRECTORY/'plan.json').read_text());old=json.loads((ROOT/'research/elastic-patch/plan.json').read_text())
 assert plan['gates']==old['gates'] and plan['cases'][:10]==old['cases'] and plan['original_case_names']==[c['name'] for c in old['cases']]
 assert len(plan['cases'])==18 and len(plan['levels'])==3
 if args.check_plan:print('Plan valid: original10 unchanged, 8 prospectively declared additions; no execution');return
 if not args.source_commit:parser.error('--source-commit required after parent freezes integration')
 source=subprocess.check_output(['git','rev-parse',args.source_commit],cwd=ROOT,text=True).strip();guard(source)
 provenance=dict(execution_source_commit=source,source_sha256={p:sha((ROOT/p).read_bytes()) for p in SOURCES},numpy_version=np.__version__,scipy_version=scipy.__version__,scope=plan['scope'],workload_scope=plan['workload_scope'])
 checkpoint=DIRECTORY/'checkpoints';p=checkpoint/'provenance.json'
 if p.exists() and json.loads(p.read_text())!=provenance:raise RuntimeError('Provenance changed; preserve old study')
 atomic(p,canonical(provenance))
 if not (DIRECTORY/'execution-source.zip').exists():
  with zipfile.ZipFile(DIRECTORY/'execution-source.zip.tmp','w',zipfile.ZIP_DEFLATED) as archive:
   for path in SOURCES:archive.write(ROOT/path,path)
  (DIRECTORY/'execution-source.zip.tmp').replace(DIRECTORY/'execution-source.zip')
 with zipfile.ZipFile(DIRECTORY/'execution-source.zip') as archive:
  assert set(archive.namelist())==set(SOURCES) and all(sha(archive.read(path))==provenance['source_sha256'][path] for path in SOURCES)
 summary=dict(**provenance,cases=[],complete=False,expected_attempts=54)
 trace_paths=[]
 def archive_progress():
  with zipfile.ZipFile(DIRECTORY/'traces.zip.tmp','w',zipfile.ZIP_DEFLATED) as archive:
   for path in trace_paths:archive.write(path,str(path.relative_to(checkpoint)))
  (DIRECTORY/'traces.zip.tmp').replace(DIRECTORY/'traces.zip')
  summary['completed_histories']=sum(a['status']=='completed' for c in summary['cases'] for a in c['runs']);summary['rejected_attempts']=sum(a['status']=='rejected' for c in summary['cases'] for a in c['runs'])
  summary['qualified_cases']=sum(c.get('qualified',False) for c in summary['cases']);summary['qualified_original_cases']=sum(c.get('qualified',False) for c in summary['cases'] if c['name'] in plan['original_case_names'])
  summary['artifact_sha256']={p:sha((DIRECTORY/p).read_bytes()) for p in ['execution-source.zip','traces.zip']}
  atomic(DIRECTORY/'summary.json',json.dumps(serial(summary),indent=2,allow_nan=False).encode()+b'\n')
 for case in plan['cases']:
  row=dict(name=case['name'],runs=[],qualified=False);summary['cases'].append(row);results=[]
  simulation={k:v for k,v in case['simulation'].items() if k not in EXTRAS}
  if 'planes' in simulation:simulation['planes']=tuple(Plane(**p) for p in simulation['planes'])
  material=Material(**case['material'])
  for level in plan['levels']:
   guard(source);settings={k:v for k,v in level.items() if k!='name'};settings['max_step']*=case['simulation'].get('step_scale',1.)
   path=checkpoint/case['name']/(level['name']+'.json');npz=path.with_suffix('.npz')
   if path.exists():attempt=json.loads(path.read_text())
   else:
    start=time.perf_counter();attempt=dict(level=level['name'],settings=settings)
    try:
     r=simulate(material,**simulation,**settings);attempt.update(status='completed',metrics=metrics(r));buffer=io.BytesIO();np.savez_compressed(buffer,**{k:v for k,v in r.items() if isinstance(v,np.ndarray)});atomic(npz,buffer.getvalue())
    except (RuntimeError,ValueError) as e:attempt.update(status='rejected',reason=str(e))
    attempt['elapsed_s']=time.perf_counter()-start;atomic(path,canonical(attempt))
   row['runs'].append(attempt)
   if attempt['status']=='completed':
    with np.load(npz) as saved:r={k:saved[k] for k in saved.files}
    results.append(r);trace_paths.append(npz)
   else:results.append(None)
   archive_progress();print(case['name'],level['name'],attempt['status'],round(attempt['elapsed_s'],3),flush=True)
  row.update(qualify(case,plan['levels'],row['runs'],results,plan['gates']))
  sim=case['simulation'];planes=simulation.get('planes',(Plane(),))
  if len(planes)==1 and not any(sim.get('gravity',[0,0,0])) and all(r is not None for r in results):
   try:
    dispatch=resolve_impact(material,planes[0].normal,sim['velocity'],sim['omega'],max_rhs_evaluations=sim.get('max_rhs_evaluations',200000),rtol=plan['levels'][-1]['rtol'],atol=plan['levels'][-1]['atol'])
    fine=results[-1];comparison=dict(method=dispatch['method'],velocity_error_m_s=float(np.linalg.norm(dispatch['velocity']-fine['states'][-1,3:6])),omega_error_rad_s=float(np.linalg.norm(dispatch['omega']-fine['states'][-1,6:9])),dissipation_error_J=float(abs(dispatch['dissipated_J']-fine['dissipated_J'][-1])),linear_impulse_error_N_s=float(np.linalg.norm(dispatch['linear_impulse_N_s']-fine['linear_impulse_N_s'][-1])),couple_impulse_error_N_m_s=float(np.linalg.norm(dispatch['independent_couple_impulse_N_m_s']-fine['couple_impulse_N_m_s'][-1])))
    comparison['passed']=comparison['velocity_error_m_s']<=plan['gates']['analytic_velocity_m_s'] and comparison['omega_error_rad_s']<=plan['gates']['analytic_omega_rad_s'] and comparison['dissipation_error_J']<=plan['gates']['energy_absolute_J']
   except (RuntimeError,ValueError) as e:comparison=dict(passed=False,rejected=str(e))
   row['dispatcher']=comparison;row['checks']['same_material_dispatcher']=bool(comparison['passed']);row['qualified'] &= bool(comparison['passed'])
  archive_progress();print('CASE',case['name'],'qualified',row['qualified'],flush=True)
 summary['complete']=True;guard(source);archive_progress();print('DONE',source,summary['qualified_original_cases'],summary['qualified_cases'],summary['completed_histories'],summary['rejected_attempts'],flush=True)
if __name__=='__main__':main()
