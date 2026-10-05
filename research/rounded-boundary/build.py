"""Build separately, retaining original and overridden precision inventories."""
import hashlib
import json
from pathlib import Path
import subprocess
from research.build_precision_backend import build,convert
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=ROOT/'build/rigid_double_rounded'
if not (D/'precision-source.json').exists():build(D,Path('/tmp/box2d-block.tar.gz'),.000001)
p=D/'source/runner.cpp';original=json.loads((D/'precision-source.json').read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest()==original['transformed']['runner.cpp']
p.write_text(convert((H/'runner.cpp').read_text()).replace('std::setprecision(10)','std::setprecision(17)'))
subprocess.run(['cmake','--build',str(D),'-j','4'],check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(H/'build-receipt.json').write_text(json.dumps({'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'original_precision_inventory_sha256':sha(D/'precision-source.json'),'prototype_sha256':sha(H/'runner.cpp'),'transformed_runner_sha256':sha(p),'binary_sha256':sha(D/'rigid_runner'),'scope':'Isolated override; original precision inventory preserved, production source and old binaries untouched.'},indent=2)+'\n')
