from pathlib import Path
import json,hashlib,subprocess,os,sys,time
p=Path(__file__).resolve().parent;r=Path.cwd();plan=json.loads((p/'23-capture-prospective-plan.json').read_text());env=dict(os.environ);env.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
import numpy as np
# This launcher performs execution only; compilation/provenance precede it separately.
results=[];outdir=p/'23-results';outdir.mkdir()
def external(d,o):
 A=np.array(d['A']);b=np.array(d['b']);x=np.array(o['p']);dep=np.array(d['dependencies']);hi=np.array(d['hi']);w=A@x-b;tol=d['tolerance_m_s'];projection=0.;cone=0.;stick=0.;support=0.;bounds=True
 for k in np.flatnonzero(dep<0):
  t,s=np.flatnonzero(dep==k);eig=np.linalg.eigvalsh(A[np.ix_([t,s],[t,s])])[-1];cap=hi[t]*x[k];z=x[[t,s]]-w[[t,s]]/eig;length=np.linalg.norm(z);factor=cap/length if length>cap and length>0 else 1.
  projection=max(projection,abs(x[k]-max(0.,x[k]-w[k]/A[k,k]))*A[k,k],np.linalg.norm(x[[t,s]]-factor*z)*eig)
  pt=np.linalg.norm(x[[t,s]]);wt=np.linalg.norm(w[[t,s]]);cone=max(cone,(pt-cap)*eig)
  if cap-pt>tol/eig:stick=max(stick,wt)
  support=max(support,abs(x[[t,s]]@w[[t,s]]+max(0.,cap)*wt));bounds&=bool(0<=x[k]<=hi[k])
 energy=float(.5*x@(w-b));scale=1+np.sum(abs(x*b));impulse_scale=max(1.,max(abs(x)));finite=bool(np.all(np.isfinite(x)) and np.all(np.isfinite(w)));ok=finite and bounds and projection<=tol and cone<=tol and stick<=tol and support<=tol*impulse_scale and energy<=tol*scale
 return dict(accepted=bool(ok),projection_m_s=float(projection),circle_violation_scaled_m_s=float(cone),sticking_m_s=float(stick),support_gap_J=float(support),passive_bound_J=energy,passivity_scale=float(scale),finite=finite,bounds=bounds,native_response_max_discrepancy=float(max(abs(w-np.array(o['w'])))))
for i,(path,h) in enumerate(plan['corpus'].items()):
 inp=r/path;assert hashlib.sha256(inp.read_bytes()).hexdigest()==h;d=json.loads(inp.read_text());row={'index':i,'input':path,'sha256':h,'variants':{}}
 for variant in ['baseline','candidate']:
  exe=p/('replay_'+variant);cmd=[str(exe),str(inp),'4096'];start=time.perf_counter();pr=subprocess.run(cmd,env=env,capture_output=True,text=True);prefix=outdir/f'{i:02d}-{variant}';prefix.with_suffix('.stdout.json').write_text(pr.stdout);prefix.with_suffix('.stderr').write_text(pr.stderr);meta={'returncode':pr.returncode,'seconds_descriptive_only':time.perf_counter()-start,'command':cmd}
  try:o=json.loads(pr.stdout);meta.update(accepted=o['accepted'],external=external(d,o),output=o)
  except Exception as e:meta['parse_error']=repr(e)
  prefix.with_suffix('.receipt.json').write_text(json.dumps({k:v for k,v in meta.items() if k!='output'},indent=2)+'\n');row['variants'][variant]=meta
 a=row['variants']['baseline'].get('output');b=row['variants']['candidate'].get('output');row['default_endpoint_and_old_counters_exact']=bool(a and b and a['p']==b['p'] and a['w']==b['w'] and a['stats']==b['stats']);row['prior22_helper_bypassed']=bool(i<22 and b and b['projection_policy']['attempts']==0)
 results.append({k:v for k,v in row.items() if k!='variants'}|{'variants':{v:{k:x for k,x in z.items() if k!='output'} for v,z in row['variants'].items()}});(outdir/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print(i,'base',None if not a else a['accepted'],'candidate',None if not b else b['accepted'],'exact',row['default_endpoint_and_old_counters_exact'],'helper',None if not b else b['projection_policy'],flush=True)
receipt={'results':results,'prior22_exact':all(x['default_endpoint_and_old_counters_exact'] and x['prior22_helper_bypassed'] and x['variants']['baseline'].get('accepted') and x['variants']['candidate'].get('accepted') for x in results[:22]),'all23_candidate_native_and_external':all(x['variants']['candidate'].get('accepted') and x['variants']['candidate'].get('external',{}).get('accepted') for x in results),'new42_baseline_decline':results[-1]['variants']['baseline'].get('accepted')==False,'no_runtime_or_world_claim':True};(outdir/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n');print({k:v for k,v in receipt.items() if k!='results'});raise SystemExit(not(receipt['prior22_exact'] and receipt['all23_candidate_native_and_external'] and receipt['new42_baseline_decline']))
