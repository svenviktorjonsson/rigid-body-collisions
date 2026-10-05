"""Independent scalar recomputation of bounded successful/failed v2 trials."""
import json,hashlib,math,zipfile,subprocess
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];D=P/'results-v2';plan=json.loads((P/'plan-v2.json').read_text());prov=json.loads((D/'provenance.json').read_text());summary=json.loads((D/'summary.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(D/'source.zip')as z:
 for name in z.namelist():assert z.read(name)==subprocess.check_output(['git','show',prov['source_commit']+':'+name],cwd=ROOT)
path=next(iter(plan['capture_sha256']));assert sha((ROOT/path).read_bytes())==plan['capture_sha256'][path];data=json.loads((ROOT/path).read_text());A=np.array(data['A']);b=np.array(data['b']);hi=np.array(data['hi']);dep=np.array(data['dependencies']);tol=data['tolerance_m_s'];records=[]
def check(raw):
 p=np.array(raw);w=np.array([math.fsum(float(a*x)for a,x in zip(row,p))-rhs for row,rhs in zip(A,b)]);error=0.
 for k in np.flatnonzero(dep<0):
  rows=np.flatnonzero(dep==k);a,d=A[rows[0],rows[0]],A[rows[1],rows[1]];eig=.5*(a+d+math.hypot(a-d,2*A[rows[0],rows[1]]));z=p[rows]-w[rows]/eig;length=math.hypot(*z);cap=hi[rows[0]]*max(0.,p[k]);projected=z if length<=cap else z*cap/length if length else z;error=max(error,abs(p[k]-max(0.,p[k]-w[k]/A[k,k]))*A[k,k],math.hypot(*(p[rows]-projected))*eig)
 energy=math.fsum(.5*float(pi)*(float(wi)-float(bi))for pi,wi,bi in zip(p,w,b));scale=1+math.fsum(abs(float(pi*bi))for pi,bi in zip(p,b));ns=np.flatnonzero(dep<0);finite=bool(np.isfinite(p).all()and np.isfinite(w).all()and np.isfinite(energy)and np.isfinite(scale));accepted=bool(finite and np.all(p[ns]>=0)and np.all(p[ns]<=hi[ns])and error<=tol and energy<=tol*scale)
 return dict(accepted=accepted,residual_m_s=error,passivity_J=energy,passivity_scale=scale)
for trial in summary['trials']:
 n=trial['native'];budget=plan['budgets'][trial['label']];assert n['svd_calls']<=budget and n['iteration_steps']<=budget and n['decline_preserves_input'];gate=check(n['returned_impulse']);last=check(n['candidate_impulse']);assert gate['accepted']==n['accepted'];assert abs(gate['residual_m_s']-trial['independent']['residual_m_s'])<1e-12
 if not n['accepted']:assert n['returned_impulse']==data['p']
 records.append(dict(label=trial['label'],svd_calls=n['svd_calls'],original_returned_gate=gate,last_numerical_candidate=last,output_unchanged_on_decline=n['decline_preserves_input']))
assert records[0]['original_returned_gate']['accepted']and not records[1]['original_returned_gate']['accepted'];assert summary['controls_passed']==9 and summary['guards_unchanged']
out=dict(passed=True,source_commit=prov['source_commit'],capture_sha256=plan['capture_sha256'][path],controls_passed=9,accepted_trials=1,rejected_trials=1,records=records,physical_law_unchanged=True,strict_original_tolerance_m_s=tol,trajectory_qualified=False)
dest=D/'independent-audit.json'
if dest.exists():assert json.loads(dest.read_text())==out
else:dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
