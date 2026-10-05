"""Recompute energy, wrench momentum, yield and refinement from frozen traces."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DIRECTORY=ROOT/'research/elastic-completion'

def sha(data):return hashlib.sha256(data).hexdigest()

def channels(case,state,strain):
 m=case['material'];R=m['radius'];mass=m['mass'];I=.4*mass*R*R;K=np.array([m['tangent_stiffness'],m['tangent_stiffness'],m['twist_stiffness']/m['effective_length']**2]);power=m['compression_exponent'];planes=case['simulation'].get('planes',[dict(normal=[0,0,1],offset=0,name='floor')]);stored=np.zeros(len(state));peak=0.
 for j,plane in enumerate(planes):
  n=np.asarray(plane['normal']);delta=np.maximum(0.,R-(state[:,:3]@n-plane.get('offset',0)));h=strain[:,3*j:3*j+3];f=np.where(delta>0,(delta/R)**power,0.)
  U=.5*f*np.sum(h*h*K,axis=1) if m['friction'] else np.zeros(len(state));stored+=.5*m['normal_stiffness']*delta*delta+U
  elastic=f[:,None]*h*K;peak=max(peak,float(np.max(np.linalg.norm(elastic,axis=1)-m['friction']*m['normal_stiffness']*delta)))
 gravity=np.asarray(case['simulation'].get('gravity',[0,0,0]));kinetic=.5*mass*np.sum(state[:,3:6]**2,axis=1)+.5*I*np.sum(state[:,6:9]**2,axis=1);potential=-mass*(state[:,:3]@gravity)
 return kinetic,potential,stored,peak

def audit():
 plan=json.loads((DIRECTORY/'plan.json').read_text());summary=json.loads((DIRECTORY/'summary.json').read_text());g=plan['gates'];source=summary['execution_source_commit']
 assert summary['complete'] and summary['expected_attempts']==54
 old=json.loads((ROOT/'research/elastic-patch/plan.json').read_text());assert plan['gates']==old['gates'] and plan['cases'][:10]==old['cases']
 for name,digest in summary['artifact_sha256'].items():assert sha((DIRECTORY/name).read_bytes())==digest
 with zipfile.ZipFile(DIRECTORY/'execution-source.zip') as sourcezip:
  assert set(sourcezip.namelist())==set(summary['source_sha256'])
  for name,digest in summary['source_sha256'].items():
   data=sourcezip.read(name);assert sha(data)==digest and data==subprocess.check_output(['git','show',f'{source}:{name}'],cwd=ROOT)
  assert sourcezip.read('research/elastic-completion/plan.json')==(DIRECTORY/'plan.json').read_bytes()
 with zipfile.ZipFile(DIRECTORY/'traces.zip') as archive:
  expected={f"{case['name']}/{level['name']}.npz" for case,row in zip(plan['cases'],summary['cases']) for level,attempt in zip(plan['levels'],row['runs']) if attempt['status']=='completed'};assert set(archive.namelist())==expected
  records=[];fine_results={};completed=0;rejected=0
  for case,row in zip(plan['cases'],summary['cases']):
   assert row['name']==case['name'];results=[];metrics=[];case_ok=True
   for level,attempt in zip(plan['levels'],row['runs']):
    checkpoint=DIRECTORY/'checkpoints'/case['name']/(level['name']+'.json');assert json.loads(checkpoint.read_text())==attempt
    if attempt['status']=='rejected':
     rejected+=1;results.append(None);case_ok=False;assert attempt['reason'];continue
    completed+=1
    with np.load(io.BytesIO(archive.read(case['name']+'/'+level['name']+'.npz'))) as saved:r={k:saved[k] for k in saved.files}
    results.append(r);assert all(np.isfinite(v).all() for v in r.values());k,p,U,peak=channels(case,r['states'],r['strain']);initial=float(k[0]+p[0]+U[0]);residual=k+p+U+r['dissipated_J']-initial;energy_limit=g['energy_absolute_J']+g['energy_relative']*abs(initial)
    np.testing.assert_allclose(r['kinetic_J'],k,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(r['potential_J'],p,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(r['stored_J'],U,rtol=1e-12,atol=1e-12)
    assert max(abs(residual))<=energy_limit and np.min(r['dissipated_J'])>=-g['energy_absolute_J'] and np.min(np.diff(r['dissipated_J']))>=-g['energy_absolute_J']
    planes=case['simulation'].get('planes',[dict(normal=[0,0,1])]);nplanes=len(planes);di=9+3*nplanes;y=r['internal_states'];ki,pi,Ui,internal_peak=channels(case,y[:,:9],y[:,9:di]);internal_residual=ki+pi+Ui+y[:,di]-initial
    assert max(abs(internal_residual))<=energy_limit and max(peak,internal_peak)<=g['yield_excess_N']
    assert np.isclose(max(peak,internal_peak),attempt['metrics']['max_yield_excess_N'],rtol=1e-8,atol=1e-8)
    mass=case['material']['mass'];I=.4*mass*case['material']['radius']**2;v0=np.asarray(case['simulation']['velocity']);w0=np.asarray(case['simulation']['omega']);gravity=np.asarray(case['simulation'].get('gravity',[0,0,0]));linear=mass*(r['states'][:,3:6]-v0-r['times'][:,None]*gravity)-r['linear_impulse_N_s'];angular=I*(r['states'][:,6:9]-w0)-r['couple_impulse_N_m_s']
    if nplanes==1:angular-=np.cross(-case['material']['radius']*np.asarray(planes[0]['normal']),r['linear_impulse_N_s'])
    else:assert np.max(abs(r['states'][:,[3,4,6,7]]))<1e-12
    assert np.max(np.linalg.norm(linear,axis=1))<1e-8 and np.max(np.linalg.norm(angular,axis=1))<1e-8
    assert attempt['metrics']['rhs_evaluations']<=case['simulation'].get('max_rhs_evaluations',200000)
    assert attempt['metrics']['separation_loss_J']<=g['separation_loss_J']
    metrics.append(dict(level=level['name'],max_energy_residual_J=float(max(abs(residual))),max_internal_energy_residual_J=float(max(abs(internal_residual))),max_yield_excess_N=max(peak,internal_peak),linear_momentum_error_N_s=float(np.max(np.linalg.norm(linear,axis=1))),angular_momentum_error_N_m_s=float(np.max(np.linalg.norm(angular,axis=1))),rhs_evaluations=attempt['metrics']['rhs_evaluations']))
   edges=[]
   if all(r is not None for r in results):
    for a,b in zip(results,results[1:]):
     assert np.array_equal(a['times'],b['times']);d=a['states']-b['states'];edge=dict(position_m=float(np.max(np.linalg.norm(d[:,:3],axis=1))),velocity_m_s=float(np.max(np.linalg.norm(d[:,3:6],axis=1))),omega_rad_s=float(np.max(np.linalg.norm(d[:,6:9],axis=1))))
     case_ok &= all(edge[key]<=g[{'position_m':'position_refinement_m','velocity_m_s':'velocity_refinement_m_s','omega_rad_s':'omega_refinement_rad_s'}[key]] for key in edge);edges.append(edge)
    assert edges==row['refinement_edges'];fine=results[-1];fine_results[case['name']]=fine;sim=case['simulation']
    if 'expected_velocity' in sim:case_ok &= np.linalg.norm(fine['states'][-1,3:6]-sim['expected_velocity'])<=g['analytic_velocity_m_s']
    if 'expected_omega' in sim:case_ok &= np.linalg.norm(fine['states'][-1,6:9]-sim['expected_omega'])<=g['analytic_omega_rad_s']
    if 'expected_dissipation' in sim:case_ok &= abs(fine['dissipated_J'][-1]-sim['expected_dissipation'])<=1e-5
    if sim.get('require_spin_reversal'):case_ok &= fine['states'][-1,8]*sim['omega'][2]<0
    lifts=[e for e in row['runs'][-1]['metrics']['events'] if e['kind']=='lift_off']
    if 'minimum_bounces' in sim:
     case_ok &= len(lifts)>=sim['minimum_bounces']
     for i,event in enumerate(lifts):
      if sim.get('bounce_pattern')=='same-floor-oblique':
       case_ok &= event['plane']=='floor' and abs(event['velocity_m_s'][0]+sim['velocity'][0]*(-1)**i)<=g['analytic_velocity_m_s'] and abs(event['omega_rad_s'][1]+sim['omega'][1]*(-1)**i)<=g['analytic_omega_rad_s'] and abs(event['omega_rad_s'][2]+sim['omega'][2]*(-1)**i)<=g['analytic_omega_rad_s']
      else:case_ok &= event['plane']==('floor' if i%2==0 else 'ceiling') and abs(event['velocity_m_s'][2]-abs(sim['velocity'][2])*(-1)**i)<=g['analytic_velocity_m_s'] and abs(event['omega_rad_s'][2]+sim['omega'][2]*(-1)**i)<=g['analytic_omega_rad_s']
     if sim.get('bounce_pattern')=='same-floor-oblique':case_ok &= metrics[-1]['max_yield_excess_N']<=1e-5+1e-8*float(np.max(fine['normal_force_N']))
   records.append(dict(name=case['name'],qualified_without_dispatcher=bool(case_ok),runs=metrics,refinement_edges=edges))
 # Replay the actual dispatchers from archived library files in a clean child
 # process. This prevents later workspace edits from changing the audit model.
 eligible=[case for case in plan['cases'] if case['name'] in fine_results and len(case['simulation'].get('planes',[0]))==1 and not any(case['simulation'].get('gravity',[0,0,0]))]
 with tempfile.TemporaryDirectory(prefix='elastic-archive-audit-') as temp:
  target=Path(temp)
  with zipfile.ZipFile(DIRECTORY/'execution-source.zip') as sourcezip:
   for name in ['research/elastic_patch.py','research/elastic_impulse.py']:
    path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(sourcezip.read(name))
  child="""import json,sys
from research.elastic_patch import Material
from research.elastic_impulse import resolve_impact
out={}
for case in json.load(sys.stdin):
 sim=case['simulation'];n=sim.get('planes',[{'normal':[0,0,1]}])[0]['normal']
 r=resolve_impact(Material(**case['material']),n,sim['velocity'],sim['omega'],max_rhs_evaluations=sim.get('max_rhs_evaluations',200000),rtol=1e-13,atol=1e-15)
 out[case['name']]={k:r[k].tolist() if hasattr(r[k],'tolist') else r[k] for k in ['method','velocity','omega','dissipated_J','linear_impulse_N_s','independent_couple_impulse_N_m_s']}
print(json.dumps(out))
"""
  dispatched=json.loads(subprocess.check_output(['python','-c',child],input=json.dumps(eligible),text=True,cwd=target))
 for record in records:
  name=record['name'];record['qualified']=record.pop('qualified_without_dispatcher')
  if name in dispatched:
   r=dispatched[name];fine=fine_results[name];errors=dict(velocity_error_m_s=float(np.linalg.norm(np.asarray(r['velocity'])-fine['states'][-1,3:6])),omega_error_rad_s=float(np.linalg.norm(np.asarray(r['omega'])-fine['states'][-1,6:9])),dissipation_error_J=float(abs(r['dissipated_J']-fine['dissipated_J'][-1])))
   record['dispatcher']=dict(method=r['method'],**errors);record['qualified'] &= errors['velocity_error_m_s']<=g['analytic_velocity_m_s'] and errors['omega_error_rad_s']<=g['analytic_omega_rad_s'] and errors['dissipation_error_J']<=g['energy_absolute_J']
 for record,row in zip(records,summary['cases']):assert bool(record['qualified'])==row['qualified']
 out=dict(execution_source_commit=source,qualified_original_cases=sum(r['qualified'] for r in records if r['name'] in plan['original_case_names']),qualified_cases=sum(r['qualified'] for r in records),completed_histories=completed,rejected_attempts=rejected,cases=records)
 assert out['qualified_cases']==summary['qualified_cases'] and out['qualified_original_cases']==summary['qualified_original_cases'] and completed==summary['completed_histories'] and rejected==summary['rejected_attempts']
 (DIRECTORY/'independent-audit.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print('Elastic completion audit PASS:',out['qualified_original_cases'],'originals;',out['qualified_cases'],'total;',completed,'histories;',rejected,'rejects')
 return out
if __name__=='__main__':audit()
