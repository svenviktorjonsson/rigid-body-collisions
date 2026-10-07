"""Portable read-only verification of frozen unsuccessful numerical trials."""
import json,hashlib,zipfile,subprocess,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];D=P/'results';plan=json.loads((P/'plan.json').read_text());summary=json.loads((D/'summary.json').read_text());prov=json.loads((D/'provenance.json').read_text());sha=lambda raw:hashlib.sha256(raw).hexdigest()
with zipfile.ZipFile(D/'source.zip')as z:
 for name in z.namelist():assert z.read(name)==subprocess.check_output(['git','show',prov['source_commit']+':'+name],cwd=ROOT)
for path,digest in plan['capture_sha256'].items():assert sha((ROOT/path).read_bytes())==digest
capture=json.loads((ROOT/next(iter(plan['capture_sha256']))).read_text());A=np.array(capture['A']);b=np.array(capture['b']);hi=np.array(capture['hi']);dep=np.array(capture['dependencies']);tol=capture['tolerance_m_s'];records=[]
for trial in summary['trials']:
 native=trial['native'];assert not native['accepted']and native['svd_calls']==native['iteration_steps']==1024 and native['decline_preserves_input'];assert np.array_equal(native['returned_impulse'],capture['p']);p=np.array(native['candidate_impulse']);w=np.array([math.fsum(float(a*x)for a,x in zip(row,p))-rhs for row,rhs in zip(A,b)]);maximum=0.
 for k in np.flatnonzero(dep<0):
  rows=np.flatnonzero(dep==k);a,d=A[rows[0],rows[0]],A[rows[1],rows[1]];eig=.5*(a+d+math.hypot(a-d,2*A[rows[0],rows[1]]));z=p[rows]-w[rows]/eig;length=math.hypot(*z);cap=hi[rows[0]]*max(0.,p[k]);projected=z if length<=cap else z*cap/length if length else z;normal=abs(p[k]-max(0.,p[k]-w[k]/A[k,k]))*A[k,k];maximum=max(maximum,normal,math.hypot(*(p[rows]-projected))*eig)
 assert maximum>tol and abs(maximum-trial['last_candidate']['residual_m_s'])<1e-12;records.append(dict(label=trial['label'],recomputed_residual_m_s=maximum,svd_calls=1024,decline_preserves_input=True))
controls=json.loads((D/'controls.stdout.json').read_text());assert controls['passed']and len(controls['controls'])==9 and all(c['passed']for c in controls['controls'])
out=dict(passed=True,source_commit=prov['source_commit'],controls_passed=9,rejected_trials_verified=4,strict_new_capture_accepted=False,records=records,no_physical_law_change=True,no_trajectory_qualification=True)
dest=D/'independent-audit.json'
if dest.exists():assert json.loads(dest.read_text())==out
else:dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
