from pathlib import Path
import json,hashlib,os
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
import numpy as np
p=Path(__file__).resolve().parent;r=Path.cwd();proof=json.loads((p/'23-results/summary.json').read_text());manifest=json.loads((p/'v2-23-ready-manifest.json').read_text());plan=json.loads((p/'v2-23-ready-plan.json').read_text());compile_receipt=json.loads((p/'23-compile-receipt.json').read_text());failures=[];checks=[]
def check(name,value):
 checks.append({'name':name,'passed':bool(value)})
 if not value:failures.append(name)
check('all_READY_sourcebytes_unchanged',all(hashlib.sha256((p/f).read_bytes()).hexdigest()==h for f,h in manifest.items()))
check('all_native_snapshot_sourcebytes_unchanged',all(hashlib.sha256((p/f).read_bytes()).hexdigest()==h for f,h in plan['native_snapshot'].items()))
check('protected_compiler_libraries_production_executables_unchanged',all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in compile_receipt['before'].items()))
check('isolated_executables_unchanged',all(hashlib.sha256((p/('replay_'+v)).read_bytes()).hexdigest()==h for v,h in compile_receipt['executables'].items()))
check('prior22exact_and_bypassed',proof['prior22_exact']);check('all23_candidate_native_and_independent_accepted',proof['all23_candidate_native_and_external']);check('new42_original_decline_retained',proof['new42_baseline_decline'])
for row in proof['results']:
 index=row['index'];inp=r/row['input'];check(f'{index}_input_unchanged',hashlib.sha256(inp.read_bytes()).hexdigest()==row['sha256']);d=json.loads(inp.read_text());A=np.array(d['A']);b=np.array(d['b']);hi=np.array(d['hi']);dep=np.array(d['dependencies']);tol=d['tolerance_m_s']
 o=json.loads((p/'23-results'/f'{index:02d}-candidate.stdout.json').read_text());x=np.array(o['p']);w=A@x-b;energy=float(.5*x@(w-b));scale=float(1+np.sum(abs(x*b)));impulse_scale=max(1.,max(abs(x)))
 check(f'{index}_strict_finite_energy_and_state',np.all(np.isfinite(x)) and np.all(np.isfinite(w)) and np.isfinite(energy) and np.isfinite(scale) and scale>0)
 check(f'{index}_strict_passivity',np.isfinite(energy) and np.isfinite(scale) and energy<=tol*scale)
 # Independent unilateral, capacity, sticking, and support-function tests.
 negative_normal=0.;negative_impulse=0.;active_normal=0.;work_complementarity=0.;circle=0.;stick=0.;support=0.;positive_friction=0.;bounds=True
 for k in np.flatnonzero(dep<0):
  t,s=np.flatnonzero(dep==k);eig=np.linalg.eigvalsh(A[np.ix_([t,s],[t,s])])[-1];pn=x[k];wn=w[k];pt=np.linalg.norm(x[[t,s]]);wt=np.linalg.norm(w[[t,s]]);cap=hi[t]*max(0.,pn);work=float(x[[t,s]]@w[[t,s]])
  negative_normal=max(negative_normal,-wn);negative_impulse=max(negative_impulse,-pn*A[k,k]);work_complementarity=max(work_complementarity,abs(pn*wn));bounds&=bool(pn<=hi[k])
  if pn*A[k,k]>tol:active_normal=max(active_normal,abs(wn))
  circle=max(circle,(pt-cap)*eig)
  if cap-pt>tol/eig:stick=max(stick,wt)
  support=max(support,abs(work+cap*wt));positive_friction=max(positive_friction,work)
 check(f'{index}_full_original_physical_laws',bounds and negative_normal<=tol and negative_impulse<=tol and active_normal<=tol and work_complementarity<=tol*impulse_scale and circle<=tol and stick<=tol and support<=tol*impulse_scale and positive_friction<=tol*impulse_scale)
 if index<22:check(f'{index}_no_extra_helper_work',o['projection_policy']['attempts']==0 and o['projection_policy']['svd_calls']==0 and o['projection_policy']['iteration_steps']==0)
 else:check('new42_fresh_finite_extra_budget',o['projection_policy']['attempts']==1 and o['projection_policy']['solves']==1 and 0<=o['projection_policy']['svd_calls']<=2048 and 0<=o['projection_policy']['iteration_steps']<=2048)
result={'passed':not failures,'checks':checks,'failures':failures,'source_scope':'Core candidate/harness/prespectiveplan published9e546715d14a5b0468f984f3b5f18eff02dfb6dd. This supplemental auditor was added locally after publication and hashed before execution; no core source mutation.','claim_limit':'23 frozen instantaneous systems only. No trajectory/refinement/physicalmaterialvalidation or runtime superiority.'};(p/'23-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print({'passed':not failures,'check_count':len(checks),'failures':failures});raise SystemExit(bool(failures))
