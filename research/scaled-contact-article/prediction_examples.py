"""Evaluate separately tabulated parameters through native2D/3D wrappers."""
from pathlib import Path
import json,numpy as np
from matrix_tools import predict_native,predict_planar
P=Path(__file__).resolve().parent;R=P.parent/'restitution-validation-report'
# Inputs are drawn from separate retained data/configuration tables, not equations.
params=json.loads((R/'rubber-surfaces/summary.json').read_text())['rows'][0]
source=json.loads((R/'rubber-surfaces/granite_a25.0_et0.49.json').read_text())
result=predict_native(source['scene'],params['normal_restitution'],params['tangential_restitution'],params['friction_hypothesis'])
err=float(np.max(abs(np.asarray(result['states'])-np.asarray(source['result']['states']))));assert err<1e-10
planar=json.loads((R/'geometry-sweep/2_box_13.0_0.3_0.2.json').read_text());c=planar['checks']
res2=predict_planar(planar['scene'],c['normal_restitution'],c['tangential_restitution'],c['mu'])
err2=float(np.max(abs(np.asarray(res2['states'])-np.asarray(planar['result']['states']))));assert err2<1e-10
(P/'prediction-examples.json').write_text(json.dumps({'3D':{'source':'rubber-surfaces/granite_a25.0_et0.49.json','parameters_from_separate_table':params['surface'],'max_reference_state_difference':err,'result':result},'2D':{'source':c['case'],'max_reference_state_difference':err2,'result':res2},'interpretation':'Reproduction of retained predictions; not new independent material validation.2D candidate remains isolated.'},indent=2)+'\n');print('Native wrapper2D/3D reproduction PASS',err,err2)
