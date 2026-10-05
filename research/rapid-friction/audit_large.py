"""Verify frozen provenance and independently audit rejected contact endpoints."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
D=H/'results-large-irregular'
def load(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
provenance=load(D/'provenance.json')
assert all(digest(Path(p))==h for p,h in provenance['guards'].items())
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
    check=namespace['external'](d,{'p':d['p'],'w':(A@impulse-b).tolist()})
    assert not check['accepted'],str(p)
    assert np.isclose(check['projection_m_s'],d['residual_m_s'],rtol=1e-5,atol=1e-12),str(p)
    rejections.append({'record':str(p.relative_to(H)),'sha256':digest(p),'rows':len(b),'independent':check})
report={'passed':True,'frozen_source_and_runtime_unchanged':True,'source':provenance['source'],'independent_rejection_count':len(rejections),'rejections':rejections}
(H/'large-provenance-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rejections'},indent=2))
