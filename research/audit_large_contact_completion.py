"""Independently recompute full frozen-system contact and energy acceptance."""
import hashlib,json
from pathlib import Path
import numpy as np


def check(capture,raw):
    data=json.loads(capture.read_text());A=np.array(data['A']);b=np.array(data['b']);dep=np.array(data['dependencies']);hi=np.array(data['hi']);p=np.array(raw,dtype=float);ns=np.flatnonzero(dep<0)
    corrections=[dict(row=int(k),original=float(p[k])) for k in ns if -1e-12<=p[k]<0]
    for c in corrections:p[c['row']]=0
    w=A@p-b;residual=0.;cone_excess=0.
    for k in ns:
        rows=np.flatnonzero(dep==k);mobility=np.linalg.eigvalsh(A[np.ix_(rows,rows)])[-1]
        normal=abs(p[k]-max(0.,p[k]-w[k]/A[k,k]))*A[k,k]
        z=p[rows]-w[rows]/mobility;cap=hi[rows[0]]*p[k];length=np.linalg.norm(z)
        projected=z*min(1.,max(0.,cap)/length) if length else z
        residual=max(residual,float(normal),float(np.linalg.norm(p[rows]-projected)*mobility))
        cone_excess=max(cone_excess,float(np.linalg.norm(p[rows])-cap))
    energy=float(.5*p@A@p-b@p);scale=float(1+np.sum(abs(p*b)));finite=bool(np.isfinite(p).all() and np.isfinite(w).all() and np.isfinite(energy))
    accepted=bool(finite and np.all(p[ns]>=0) and np.all(p[ns]<=hi[ns]) and residual<=data['tolerance_m_s'] and energy<=data['tolerance_m_s']*scale)
    return dict(capture=str(capture),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),rows=len(p),independent_full_original_residual_m_s=residual,
                passive_change_bound_J=energy,passivity_scale=scale,maximum_cone_excess_N_s=cone_excess,normal_roundoff_clamps=corrections,
                finite=finite,all_normal_impulses_nonnegative=bool(np.all(p[ns]>=0)),accepted=accepted,p=p.tolist())


def run():
    root=Path('research/completion-large-contact-review');records=[]
    for filename,member in [('reference_2-fb-trf-normal-qp-10-subset.json',None),('reference_0-sticking-mode-trials.json',0)]:
        source=root/filename;package=json.loads(source.read_text());trial=package if member is None else package['trials'][member]
        record=check(Path(package['capture']),trial['p']);record.update(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),candidate='independent-scipy');records.append(record)
    native=Path('research/coulomb-normal/active-v3-four-native.jsonl')
    for line in native.read_text().splitlines():
        trial=json.loads(line)
        if trial['accepted'] and 'fast_rotate_shake27_hulls7301' in trial['capture']:
            record=check(Path(trial['capture']),trial['p']);record.update(source=str(native),source_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),candidate='native-active-face-v3');records.append(record)
    assert len(records)==4 and all(r['accepted'] for r in records)
    (root/'independent-final-audit.json').write_text(json.dumps(dict(schema='independent-large-capture-final-audit-v1',cases=records,scope='Acceptance of retained final frozen captures only; not acceptance of later trajectories, performance superiority, or experimental material calibration.'),indent=2)+'\n')
    print('Independent full-law and passivity audit PASS: four candidates, 231/321 rows')


def audit_final_native():
    source=Path('research/coulomb-normal/active-final-four-native.jsonl');records=[]
    for line in source.read_text().splitlines():
        trial=json.loads(line);record=check(Path(trial['capture']),trial['p'])
        record.update(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),candidate='native-final-active-face',svd_calls=trial['svd_calls'])
        records.append(record)
    assert len(records)==4 and all(r['accepted'] for r in records)
    Path('research/completion-large-contact-review/independent-final-native-audit.json').write_text(json.dumps(dict(schema='independent-final-native-active-face-audit-v1',cases=records,scope='Four final captured systems; no qualification of subsequent full trajectories.'),indent=2)+'\n')
    for r in records:print(r['capture'],r['independent_full_original_residual_m_s'],r['passive_change_bound_J'],r['svd_calls'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--final-native',action='store_true');args=parser.parse_args()
    audit_final_native() if args.final_native else run()
