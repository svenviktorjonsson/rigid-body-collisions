"""Render descriptive retained costs separately from qualified speed ratios."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
D=H/'results-large-irregular'
summary=json.loads((D/'summary.json').read_text())
scenes=json.loads((D/'scenes.json').read_text())
lines=['# Larger irregular-shape rapid-friction cases','',
'All cases retain friction0.4, gravity9.81m/s², +/-20m/s reversals at0.04/0.08s and0.12s simulated. Counts exclude the container. Shapes rotate freely; prescribed containers translate. Planar shapes use seed7301; spatial hulls seed42.','',
'2D mixed populations contain one-third concave triangular compounds and two-thirds irregular convex polygons. 3D hulls use12 asymmetric random vertices within0.1m support radius. These hulls are convex; this suite does not test concave3D compounds. Planar cell spacing is0.24m; spatial spacing0.205m. Disk scaling controls therefore do not isolate body count from the earlier qualified disk packing.','',
'Planar execution uses supported experimental Float64 Box2D with1um numerical slop,128 velocity and12 position iterations, authored0.01m collision skin. Spatial execution uses production circular Coulomb law,4096 iterations and original bounded recovery. Defaults and accuracy gates are unchanged.','',
'Five predeclared reference levels are10,5,2.5,1.25,0.625us. Qualification requires both adjacent quarter-budget edges; only qualified scenes receive candidate selection and three alternating timed repetitions. Failed histories/rejected systems remain archived.','',
'| Case | Bodies | Complete reference histories | Native costs by level (s) | Process elapsed by level (s) | Reference qualified | Timed gain |',
'|---|---:|---:|---|---|---|---|']
for name,item in summary.items():
    costs=[];elapsed=[];complete=0
    for i in range(item['reference_levels_executed']):
        r=json.loads((D/name/f'reference_{i}.json').read_text())
        elapsed.append(f"{r['process_elapsed_s']:.3f}");complete+=r['complete'];costs.append(f"{r['result']['step_s']:.3f}" if r['complete'] else 'rejected')
    b=item.get('benchmark');gain=f"{b['reference_over_candidate']:.2f}×" if b and b['qualified'] else 'unqualified'
    lines.append(f"| {name} | {len(scenes[name]['scene']['bodies'])-1} | {complete}/{item['reference_levels_executed']} | {', '.join(costs)} | {', '.join(elapsed)} | {item['reference_qualified']} | {gain} |")
lines+=['','Single reference costs are descriptive, not repeated benchmark medians. Rejected runs cover only their accepted prefix: their process elapsed is time to failure, not a full-horizon timing. No speed ratio is assigned to an unqualified reference. Native2D times physics/observer excluding frame output/diagnostics; native3D includes state recording. Compare within a case. Process elapsed and exact numerical settings are saved in each record. External host load is uncontrolled.','',
'Reproduce in a fresh output directory with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m research.rapid-friction.large_irregular`; source/binary/runtime hashes are in provenance.json. Portable archive audit: `python -m research.rapid-friction.audit`.','',
'This expands the external synthetic benchmark coverage; it does not establish material calibration, a VKF port or general irregular-shape qualification.']
(H/'LARGE-IRREGULAR-RESULTS.md').write_text('\n'.join(lines)+'\n')
