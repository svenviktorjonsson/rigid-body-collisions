"""Verify frozen provenance and independently audit rejected contact endpoints."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
D=H/'results-large-irregular'
def load(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
provenance=load(D/'provenance.json')
ROOT=H.parents[1]
snapshots=load(H/'reference-binary-snapshots.json') if (H/'reference-binary-snapshots.json').exists() else {}
live_unchanged=True
for name,sha in provenance['guards'].items():
    path=Path(name);current=path.exists() and digest(path)==sha
    live_unchanged &= current
    if name in snapshots:
        assert snapshots[name]['sha256']==sha and digest(snapshots[name]['snapshot'])==sha
    else:
        relative=path.relative_to(ROOT)
        archived=subprocess.check_output(['git','show',provenance['source']+':'+str(relative)],cwd=ROOT)
        assert hashlib.sha256(archived).hexdigest()==sha
assert all(digest(Path(p))==h for libraries in provenance['runtime'].values() for p,h in libraries.items())
assert load(D/'final.json')['complete']
source=H.parent/'new-combined-contact-review/run-20261005T194211Z/run23.py'
function=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external')
namespace={'np':np}
exec(compile(ast.Module(body=[function],type_ignores=[]),'archived-independent-law','exec'),namespace)
rejections=[]
for p in sorted(D.rglob('*.rejection.json')):
    d=load(p);A=np.array(d['A']);b=np.array(d['b']);impulse=np.array(d['p'])
    assert A.shape==(len(b),len(b)) and impulse.shape==b.shape
    assert np.isfinite(A).all() and np.isfinite(b).all()
    if d['phase']=='position_translation':
        response=A@impulse-b;diagonal=np.diag(A)
        projected=np.clip(impulse-response/diagonal,d['lo'],d['hi'])
        residual=float(np.max(np.abs(impulse-projected)*diagonal))
        check={'accepted':bool(np.isfinite(response).all() and residual<=d['tolerance_m_s']), 'projection_m_s':residual,'phase':'position_translation'}
    else:
        check=namespace['external'](d,{'p':d['p'],'w':(A@impulse-b).tolist()})
    assert not check['accepted'],str(p)
    assert np.isclose(check['projection_m_s'],d['residual_m_s'],rtol=1e-5,atol=1e-12),str(p)
    rejections.append({'record':str(p.relative_to(H)),'sha256':digest(p),'rows':len(b),'independent':check})
report={'passed':True,'archived_source_binary_and_runtime_hashes_verified':True,'current_live_source_and_binary_unchanged':bool(live_unchanged),'source':provenance['source'],'independent_rejection_count':len(rejections),'rejections':rejections}
(H/'large-provenance-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rejections'},indent=2))
