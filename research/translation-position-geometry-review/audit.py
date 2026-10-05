"""Independent archived geometry source and actual translation Gram audit."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
import numpy as np
P=Path(__file__).parent;ROOT=P.parents[1];D=P/'results';destination=D/'independent-geometry-audit.json';assert not destination.exists();prov=json.loads((D/'provenance.json').read_text());receipt=json.loads((D/'receipt.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(D/'execution-source.zip')as z:
 assert set(z.namelist())==set(prov['source_hashes'])
 for p,expected in prov['source_hashes'].items():
  raw=z.read(p);assert sha(raw)==expected;assert raw==subprocess.check_output(['git','show',prov['execution_source_commit']+':'+p],cwd=ROOT)
capture=D/'rejected-normal-system.json';geometry=D/'rejected-normal-system.json.geometry.json';assert sha(capture.read_bytes())==receipt['capture_sha256']=='c26bc4e45377fdd7fa069b41f355065c6deed9d112de0fdb58fd26c173a3d617';assert sha(geometry.read_bytes())==receipt['geometry_sha256']=='ee20bdc321693a9dc929b6bd2cde5ad16a8b06a1d18f1f1958ee1ff76c0fa775'
c=json.loads(capture.read_text());g=json.loads(geometry.read_text());body={v['solver_body_id']:v for v in g['bodies']};n=len(c['b']);assert n==len(g['rows'])==74;maxerr=0.
for i,a in enumerate(g['rows']):
 assert a['split_target_m_s']==c['b'][i]
 for j,b in enumerate(g['rows']):
  response=0.
  for s in ['a','b']:
   for t in ['a','b']:
    ident=a['solver_body_id_'+s]
    if ident==b['solver_body_id_'+t]:response+=body[ident]['inverse_mass']*np.dot(a['linear_jacobian_'+s],b['linear_jacobian_'+t])
  maxerr=max(maxerr,abs(response-c['A'][i][j]))
assert maxerr<1e-12
out=dict(passed=True,execution_source_commit=prov['execution_source_commit'],capture_sha256=receipt['capture_sha256'],geometry_sha256=receipt['geometry_sha256'],all74_translation_Gram_reconstructed_max_error=maxerr,original_capture_byte_identical=True,original_translation_rejection_retained=True,support_rows=[{k:g['rows'][i][k]for k in ['normal_row_index','body_id_a','body_id_b','normal_world_on_b','signed_distance_m','split_target_m_s']}for i in [11,54,55]],scope='Geometry/provenance audit only; no changed-policy or full-trajectory qualification')
destination.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
