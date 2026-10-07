"""Independent scalar/rotation and provenance audit of native cache controls."""
import hashlib,json,itertools,subprocess,zipfile,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent;ROOT=P.parents[1];D=P/'results'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
plan=json.loads((P/'plan.json').read_text());fixture=json.loads((P/'fixture.json').read_text());prov=json.loads((D/'provenance.json').read_text());result=json.loads((D/'controls.json').read_text());receipt=json.loads((D/'receipt.json').read_text())
assert receipt['passed'] and result['passed'] and result['controls']==plan['expected_control_count']==8
with zipfile.ZipFile(D/'source.zip')as z:
 for path,expected in prov['source_sha256'].items():
  raw=z.read(path);assert sha(raw)==expected;assert raw==subprocess.check_output(['git','show',prov['execution_source_commit']+':'+path],cwd=ROOT)
geometries={g['id']:np.array(g['vertices'],dtype=float)for g in fixture['geometries']};maxerror=0.;count=0;keys=set()
for trial in result['trials']:
 assert trial['passed'];margin=trial['declared_margin_m'];rotated=trial['child_transform_rotated'];key=(trial['geometry'],margin,rotated);assert key not in keys;keys.add(key)
 vertices=geometries[trial['geometry']];R=Rotation.from_rotvec(.37*np.array([1,2,3])/math.sqrt(14)).as_matrix()if rotated else np.eye(3);offset=np.array([.014,-.003,.006])if rotated else np.zeros(3)
 actual=vertices@R.T+offset;supportlo=actual.min(axis=0)-margin;supporthi=actual.max(axis=0)+margin
 for k,expected in [('min',supportlo),('max',supporthi)]:maxerror=max(maxerror,float(np.max(abs(np.array(trial['actual_support_aabb'][k])-expected))))
 bounds=[]
 for variant,cachedmargin in [('old',.04),('corrected',margin)]:
  lo=vertices.min(axis=0)-cachedmargin-margin;hi=vertices.max(axis=0)+cachedmargin+margin;corners=np.array(list(itertools.product(*zip(lo,hi))))@R.T+offset;expectedlo=corners.min(axis=0);expectedhi=corners.max(axis=0);reported=trial[variant+'_aabb'];blo=np.array(reported['min']);bhi=np.array(reported['max']);bounds.append((blo,bhi))
  maxerror=max(maxerror,float(np.max(abs(blo-expectedlo))),float(np.max(abs(bhi-expectedhi))))
  threshold=.02*(.5*np.linalg.norm(bhi-blo)+np.linalg.norm(.5*(bhi+blo)));maxerror=max(maxerror,abs(threshold-trial[variant+'_compound_breaking_threshold_m']))
 old,new=bounds;assert np.all(old[0]<new[0]-.01)and np.all(old[1]>new[1]+.01);assert np.all(new[0]<=supportlo+1e-12)and np.all(new[1]>=supporthi-1e-12)
 assert trial['native_support_error_m']<1e-12 and trial['compound_formula_error_m']<1e-12
 if not rotated and margin==0:assert trial['zero_margin_identity_exact_vertex_bounds']
 count+=1
assert count==8 and maxerror<1e-12
out=dict(passed=True,controls_independently_recomputed=count,maximum_scalar_geometry_error_m=maxerror,source_commit=prov['execution_source_commit'],source_archive_sha256=sha((D/'source.zip').read_bytes()),controls_sha256=sha((D/'controls.json').read_bytes()),trajectory_cause_proven=False,trajectory_qualified=False,scope='Independently recomputed cached bounding boxes, support, and threshold formulas; native support and compound diagnostics retained, no world simulation.')
dest=D/'independent-audit.json'
if dest.exists():assert json.loads(dest.read_text())==out
else:dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
