"""Bounded active-normal subset + stick/slide enumeration, exact final law."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import importlib.util,itertools
spec=importlib.util.spec_from_file_location('active_modes',Path('research/completion-review/active_modes.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
source=Path('research/hull-active-completion/results/rejections/fast_shake8_hulls42/reference_1.json');data=json.loads(source.read_text());directory=Path(__file__).parent
for release in [(4,),(3,),(2,),(3,4),(2,4),(2,3),(2,3,4)]:
 name='ref1-release-'+'-'.join(map(str,release));trial=dict(data);p=np.asarray(data['p']).copy();dep=np.asarray(data['dependencies'])
 for k in release:p[k]=0;p[dep==k]=0
 trial['p']=p.tolist();capture=directory/(name+'-initial.json');capture.write_text(json.dumps(trial)+'\n')
 start=time.perf_counter();result=module.examine(capture,starts=3);result['source_capture']=str(source);result['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();result['released_normals']=list(release);result['elapsed_s']=time.perf_counter()-start
 dest=directory/(name+'.json');assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(name,result['found_exact_solution'],result['best']['residual_m_s'],result['elapsed_s'],flush=True)
 if result['found_exact_solution']:break
